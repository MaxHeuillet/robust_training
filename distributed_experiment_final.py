import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from comet_ml import Experiment

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import torch.multiprocessing as mp 
from multiprocessing import Queue 


from autoattack import AutoAttack
from ray.air import session  

from utils import Setup, Hp_opt, move_dataset_to_tmpdir, move_architecture_to_tmpdir
from databases import load_data2
from architectures import load_architecture, CustomModel
from losses import get_loss, get_eval_loss
from utils import get_args2, set_seeds, load_optimizer
# from utils import ActivationTrackerAggregated, register_hooks_aggregated, compute_stats_aggregated

from hydra import initialize, compose
from omegaconf import OmegaConf

from ray import train
import sys
import ray
import shutil
from pathlib import Path  # In case config.dataset_path is a string


def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2).item()
            # if torch.isnan(param_norm):
            #     print(f"Gradient contains NaN : {p}")
            # elif torch.isinf(param_norm):
            #     print(f"Gradient contains Inf : {p}")
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def check_for_nans(tensors, tensor_names):
    for tensor, name in zip(tensors, tensor_names):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaNs!")

def get_config_id(cfg) -> str:
    # Join the values into a string
    serialized_values = cfg.backbone + '_' + cfg.dataset + '_' + cfg.loss_function
    print('serialized_values', serialized_values)
    return serialized_values

class BaseExperiment:

    def __init__(self, setup, base_config):

        self.setup = setup
        self.base_config = base_config

    def initialize_logger(self, rank, config):

        logger = None
        
        if rank == 0:
            logger = Experiment(api_key="I5AiXfuD0TVuSz5UOtujrUM9i",
                                    project_name=config.project_name,
                                    workspace="maxheuillet",
                                    auto_metric_logging=False,
                                    auto_output_logging=False)
            
            logger.set_name( config.exp_id )
            
            logger.log_parameter("run_id", os.getenv('SLURM_JOB_ID') )
            logger.log_parameter("global_process_rank", rank)
            logger.log_parameters(config)
        
        return logger
        
    def initialize_loaders(self, config, rank, ):

        # train_dataset, val_dataset, test_dataset, N = load_data(config, common_corruption)
        train_dataset, val_dataset, _, _, N = load_data2(config)

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.setup.world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.setup.world_size, rank=rank, drop_last=True)
  
        nb_workers = 3 if torch.cuda.device_count() > 1 else 1

        print('initialize dataoader', rank,flush=True) 
        trainloader = DataLoader(train_dataset, 
                                    batch_size=self.setup.train_batch_size(config), 
                                    sampler=train_sampler, 
                                    num_workers=nb_workers, 
                                    pin_memory=True) 
        
        valloader = DataLoader(val_dataset, 
                                    batch_size=int( 0.65 * self.setup.train_batch_size(config) ), 
                                    sampler=val_sampler, 
                                    num_workers=nb_workers,
                                    pin_memory=True)
    
        
        return trainloader, valloader, train_sampler, val_sampler, N
    
    def initialize_loaders_test(self, config, rank,):

        _, _, test_dataset, common_dataset, N = load_data2(config)
        nb_workers = 3 

        world_size = self.setup.world_size if not self.setup.hp_opt else 4

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        common_sampler = DistributedSampler(common_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        
        
        
        testloader = DataLoader(test_dataset, 
                                    batch_size=self.setup.test_batch_size(config), 
                                    sampler=test_sampler, 
                                    num_workers=nb_workers,
                                    pin_memory=True)
        
        commonloader = DataLoader(common_dataset, 
                                    batch_size=self.setup.test_batch_size(config), 
                                    sampler=common_sampler, 
                                    num_workers=nb_workers,
                                    pin_memory=True)
        
        return testloader, commonloader, test_sampler, common_sampler, N

    def training(self, config, rank=None ):

        if self.setup.hp_opt: # we have to merge here because HP is a dictionary with sample objects, which is not compatible with Omegaconf
            base_config = self.base_config
            config = OmegaConf.merge(base_config, config)
            rank = train.get_context().get_world_rank()
            logger = None
            resources = session.get_trial_resources()
            print(f"Trial resource allocation: {resources}")
            
        else:
            self.setup.distributed_setup(rank)
            logger = self.initialize_logger(rank, config)

        print('### load model', flush=True)

        trainloader, valloader, train_sampler, val_sampler, N = self.initialize_loaders(config, rank)

        print('### load model', flush=True)

        model = load_architecture(config, N, )

        model = CustomModel(config, model, )
        model.set_fine_tuning_strategy()
        model.to(rank)
        model = DDP(model, device_ids=[rank])

        # load optmizer after model
        optimizer = load_optimizer(config, model,)  
        
        # print('start the loop')
        
        scheduler = CosineAnnealingLR( optimizer, T_max=config.epochs, eta_min=0 ) if config.scheduler else None

        self.fit(config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger)

        dist.barrier() 

        if not self.setup.hp_opt and rank == 0:

            src = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
            model_name = src / f"{config.exp_id}.pt"
            model_to_save = model.module
            model_to_save = model_to_save.cpu()
            torch.save(model_to_save.state_dict(), str(model_name) )
            print('Model saved by rank 0')

            dest = Path(config.trained_statedicts_path).expanduser().resolve() / config.project_name 
            os.makedirs(str(dest), exist_ok=True) 

            shutil.copy2(str(model_name), str(dest))
            print(f"✅ Moved successfully.")
            
            logger.end()
        
        self.setup.cleanup()
        print('processes ended', flush=True)
        return True 

    def fit(self, config, model, optimizer, scheduler, trainloader, valloader, train_sampler, val_sampler, N, rank, logger=None):

        # Gradient accumulation:
        effective_batch_size = 1024
        loss_scale = 0.50 if config.loss_function == 'TRADES_v2' else 1.00
        per_gpu_batch_size = int( self.setup.train_batch_size(config) * loss_scale ) # Choose a batch size that fits in memory
        accumulation_steps = max(1, effective_batch_size // (self.setup.world_size * per_gpu_batch_size))
        global_step = 0  # Track global iterations across accumulation steps
        print('effective batch size', effective_batch_size, 'per_gpu_batch_size', per_gpu_batch_size, 'accumulation steps', accumulation_steps)

        scaler = GradScaler() 
         
        model.train()

        print('epochs', config.epochs)
        batch_step = 0
        update_step = 0
        
        for iteration in range(1, config.epochs+1):

            train_sampler.set_epoch(iteration)
            val_sampler.set_epoch(iteration)

            # print('start batches')

            for batch_id, batch in enumerate( trainloader ) :

                data, target = batch

                data, target = data.to(rank), target.to(rank) 

                # Use autocast for mixed precision during forward pass

                try:
                    with autocast():
                        loss_values, logits = get_loss(config, model, data, target)
                        loss = loss_values.mean() / accumulation_steps
                    if not torch.isfinite(loss):
                        raise ValueError(f"Loss is not finite: {loss.item()}")

                    scaler.scale(loss).backward()
                except (RuntimeError, ValueError) as e:
                    if "out of memory" in str(e).lower():
                        print("⚠️ Caught OOM during backward")
                        torch.cuda.empty_cache()
                    else:
                        print(f"⚠️ Caught exception: {e}")
                    if self.setup.hp_opt:
                        session.report({"loss": float("inf")})
                    sys.exit(0)  # prevent Ray from hanging
                
                global_step += 1

                if not self.setup.hp_opt and rank == 0:
                    metrics = { "global_step": global_step, 
                                "loss_value": loss.item() * accumulation_steps, }
                    logger.log_metrics(metrics, epoch=iteration, step = batch_step )

                batch_step += 1

                if (batch_id + 1) % max(1, accumulation_steps) == 0 or (batch_id + 1) == len(trainloader):

                    if not self.setup.hp_opt and rank == 0:
                        # print('unscale', rank, flush=True)
                        scaler.unscale_(optimizer)
                            
                        gradient_norm = compute_gradient_norms(model)
                            
                        metrics = { "gradient_norm": float(gradient_norm),   }
                            
                        logger.log_metrics(metrics, epoch=iteration, step=update_step, )


                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after optimizer step

                    update_step += 1

            if self.setup.hp_opt:
                self.validation( config, valloader, model, logger, iteration, rank)
            
            # elif not self.setup.hp_opt and iteration in [10, 40]: #25,
            #     self.validation( valloader, model, logger, iteration, rank)
                
            if scheduler is not None: scheduler.step()

            print(f'Rank {rank}, Iteration {iteration},', flush=True) 

            break


    def hyperparameter_optimization(self, config):  

        self.setup.hp_opt = True 

        # Check if experiment path exists and delete it before starting a new run
        hpo_opt_path = Path(config.hpo_path).expanduser().resolve() #os.path.abspath(os.path.expandvars(os.path.expanduser(config.hpo_path)))
        experiment_path = hpo_opt_path / config.project_name
        os.makedirs(experiment_path, exist_ok=True)
        existing_experiment_path = hpo_opt_path / config.project_name / config.exp_id
        if os.path.exists(existing_experiment_path):
            print(f"Deleting existing experiment directory: {existing_experiment_path}")
            shutil.rmtree(existing_experiment_path)

        tmp_dir = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
        os.environ["RAY_TMPDIR"] = str(tmp_dir)
        os.environ["RAY_STARTUP_TIMEOUT_SECONDS"] = "120"
        os.environ["RAY_GCS_SERVER_REQUEST_TIMEOUT_SECONDS"] = "60"
        os.environ["RAY_WORKER_REGISTER_TIMEOUT_SECONDS"] = "60"
        os.environ["RAY_NUM_HEARTBEATS_TIMEOUT"] = "120"
        
        print('initialize ray')
        ray.init( _temp_dir=str(tmp_dir) ,
                 include_dashboard=False, 
                 logging_level="DEBUG")
        
        print('end initialize')

        hp_search = Hp_opt(config, )
        print('initialized hp search')

        tuner = hp_search.get_tuner( self.training )
        print('initialized tuner')

        result_grid = tuner.fit()
        print('finished grid')
        
        best_result = result_grid.get_best_result()
        print("Best hyperparameters found were: ", best_result.config)

        self.setup.hp_opt = False

        #### Save the optimal configuration:
        directory_path = Path(config.configs_path) / "HPO_results" / config.project_name
        os.makedirs(directory_path, exist_ok=True)
        optimal_config = OmegaConf.merge(config, best_result.config['train_loop_config']  )
        output_path = directory_path / f"{config.exp_id}.yaml"
        OmegaConf.save(optimal_config, output_path)

        src = Path(os.path.expandvars(config.work_path)).expanduser().resolve() / config.exp_id
        dest = Path(config.hpo_path).expanduser().resolve() / config.project_name / config.exp_id
        print(f"📦 Moving HPO results from {src} to {dest}")
        shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
        print(f"✅ Moved successfully.")

        ray.shutdown()


    def validation(self, config, valloader, model, logger, iteration, rank):

        total_loss, total_correct_nat, total_correct_adv, total_examples, _, _ = self.validation_loop(config, valloader, model, rank)

        dist.barrier() 
        val_loss, _, _ = self.setup.sync_value(total_loss, total_examples, rank)
        nat_acc, _, _ = self.setup.sync_value(total_correct_nat, total_examples, rank)
        adv_acc, _, _ = self.setup.sync_value(total_correct_adv, total_examples, rank)

        if self.setup.hp_opt:
            print('val loss', val_loss)
            session.report({"loss": val_loss})

        elif not self.setup.hp_opt and rank == 0:

            print('hey', flush=True)

            metrics = { "val_loss": val_loss, "val_nat_accuracy": nat_acc, "val_adv_accuracy": adv_acc, }
                        #"zero_adv_val": adv_zero, "dormant_adv_val": adv_dormant, "overactive_adv_val": adv_overact,
                
            logger.log_metrics(metrics, epoch=iteration)

    def validation_loop(self, config, valloader, model, rank):

        model.eval()

        total_loss = 0.0
        total_correct_nat = 0
        total_correct_adv = 0
        total_examples = 0

        for batch_id, batch in enumerate( valloader ):

            data, target = batch

            data, target = data.to(rank), target.to(rank) 
            # print('shape validation batch', data.shape)
                
            with torch.autocast(device_type='cuda'):
                loss_values, logits_nat, logits_adv = get_eval_loss(config, model, data, target, )

            if not torch.isfinite(loss_values).all():
                print(f"⚠️ NaN detected in validation loss at batch {batch_id}")
                return float("inf"), 0, 0, 1, None, None

            total_loss += loss_values.sum().item()

            preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
            preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

            # Accumulate correct predictions
            total_correct_nat += (preds_nat == target).sum().item()
            total_correct_adv += (preds_adv == target).sum().item()
            total_examples += target.size(0)

            if batch_id== 2:
                break

        return total_loss, total_correct_nat, total_correct_adv, total_examples, None, None
    

    def test(self, rank, result_queue, corruption_type, config):

        if corruption_type == 'common':
            _, testloader, _, _, N = self.initialize_loaders_test(config, rank)
        else:
            testloader, _, _, _, N = self.initialize_loaders_test(config, rank)

        # test_sampler.set_epoch(0)  
        print('dataloader', flush=True) 
        
        model = load_architecture(config, N, )

        model = CustomModel(config, model, )
        #model.set_fine_tuning_strategy()

        path = os.path.expanduser(config.trained_statedicts_path)
        src =  os.path.join(path, config.project_name, f"{config.exp_id}.pt")
        dest = Path(os.path.expandvars(config.work_path)).expanduser().resolve()
        shutil.copy2(str(src), str(dest))
        load_path = os.path.join(dest, f"{config.exp_id}.pt")

        trained_state_dict = torch.load(load_path, weights_only=True, map_location='cpu')
        model.load_state_dict(trained_state_dict)
        model.to(rank)

        model.eval()

        print('start test loop', flush=True) 
        stats_nat, stats_adv = self.test_loop(testloader, config, model, N, rank, corruption_type)
        print('end test loop', flush=True) 

        result_queue.put( (rank, stats_nat, stats_adv) )
        print(f"Rank {rank}: Results sent to queue", flush=True)
        
    def test_loop(self, testloader, config, model, N, rank, corruption_type):

        def forward_pass(x):
            return model(x)
        
        device = torch.device(f"cuda:{rank}")

        if corruption_type in ['Linf' , 'L2' , 'L1']:

            nb_correct_nat = 0
            nb_correct_adv = 0
            nb_examples = 0
            print('stats', nb_correct_nat, nb_correct_adv, nb_examples, flush=True)
            if corruption_type == 'Linf':
                distance = config.epsilon
            elif corruption_type == 'L2':    
                distance = 2.0
            elif corruption_type == 'L1':
                distance = 75.0
            else:
                distance = None
                print('not implemented error in the distance', flush=True)

            adversary = AutoAttack(forward_pass, norm=corruption_type, eps=distance, version='standard', verbose = False, device = device)
            print('adversary instanciated', flush=True) 
            
            for _, batch in enumerate( testloader ):

                x_nat, target = batch

                x_nat, target = x_nat.to(rank), target.to(rank) 

                batch_size = x_nat.size(0)

                print('start batch iterations', rank, _,batch_size, len(testloader), flush=True) 

                x_adv = adversary.run_standard_evaluation(x_nat, target, bs = batch_size )

                logits_nat, logits_adv = model(x_nat, x_adv)

                preds_nat = torch.argmax(logits_nat, dim=1)
                preds_adv = torch.argmax(logits_adv, dim=1)

                nb_correct_nat += (preds_nat == target).sum().item()
                nb_correct_adv += (preds_adv == target).sum().item()
                nb_examples += target.size(0)

                if _ == 2:
                    break

            stats_nat = { 'nb_correct':nb_correct_nat, 'nb_examples':nb_examples }
            stats_adv = { 'nb_correct':nb_correct_adv, 'nb_examples':nb_examples }

        elif corruption_type == 'common':

            nb_correct_adv = 0
            nb_examples = 0
            
            for _, batch in enumerate( testloader ):

                x_adv, target = batch

                x_adv, target = x_adv.to(rank), target.to(rank) 

                batch_size = x_adv.size(0)

                logits_nat, logits_adv = model(x_adv, x_adv)

                preds_adv = torch.argmax(logits_adv, dim=1)

                nb_correct_adv += (preds_adv == target).sum().item()
                nb_examples += target.size(0)

                if _ == 2:
                    break
            
            stats_nat = { 'nb_correct':None, 'nb_examples':None }
            stats_adv = { 'nb_correct':nb_correct_adv, 'nb_examples':nb_examples }

        else:
            print("not implemented error")
        
        return stats_nat, stats_adv
    
    def launch_test(self, corruption_type, config):
        # import psutil  # Requires: pip install psutil
        # Create a Queue to gather results

        result_queue = Queue()

        # Launch evaluation processes
        processes = []

        for rank in range(self.setup.world_size): # 

            p = mp.Process(target=self.test, args=(rank, result_queue, corruption_type, config))
            p.start()

            # print(f"Process {p.pid} assigned to cores: {core_groups[rank]}", flush=True)
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Gather results from the Queue
        statistics_nat = {}
        statistics_adv = {}
        while not result_queue.empty():
            rank, stats_nat, stats_adv = result_queue.get()
            statistics_nat[rank] = stats_nat
            statistics_adv[rank] = stats_adv

        print('result statistics', statistics_nat, statistics_adv, flush=True)

        # Log the aggregated results
        if corruption_type == 'Linf':
            statistic = self.setup.aggregate_results(statistics_nat, 'clean')
            self.setup.log_results(config, statistic)
            statistic = self.setup.aggregate_results(statistics_adv, corruption_type)
            self.setup.log_results(config, statistic)
        else:
            statistic = self.setup.aggregate_results(statistics_adv, corruption_type)
            self.setup.log_results(config, statistic)
        

def training_wrapper(rank, experiment, config ):
    print('we are launching training')
    experiment.training(config, rank=rank)
    print('we are done training')
    return True

# @hydra.main(config_path="./configs", version_base=None)
def main():

    initialize(config_path="./configs", version_base=None)

    # The rest is your existing init:
    args_dict = get_args2()
    if 'linearprobe_50epochs' in args_dict['project_name']:
        local_config = compose(config_name="default_config_linearprobe50")
    elif 'full_fine_tuning_5epochs' in args_dict['project_name']:
        local_config = compose(config_name="default_config_fullfinetuning5")
    elif 'full_fine_tuning_50epochs' in args_dict['project_name']:
        local_config = compose(config_name="default_config_fullfinetuning50")
    else:
        print('error in the experiment name', flush=True)
        sys.exit(1)

    mode = args_dict['mode']
    args_dict.pop("mode")

    config_base = OmegaConf.merge(local_config, args_dict)
    config_base.exp_id = get_config_id(config_base)

    set_seeds(config_base.seed)

    move_dataset_to_tmpdir(config_base)
    move_architecture_to_tmpdir(config_base)


    world_size = torch.cuda.device_count()
    setup = Setup(world_size)
    experiment = BaseExperiment(setup, config_base)

    if mode == "hpo":
        print("HPO step", flush=True)
        experiment.hyperparameter_optimization(config_base)

    elif mode == "train":
        torch.multiprocessing.set_start_method("spawn", force=True)
        print("Training step", flush=True)
        # load best config from HPO results
        path = os.path.join(config_base.configs_path, "HPO_results",
                            config_base.project_name, f"{config_base.exp_id}.yaml")
        config_optimal = OmegaConf.load(path)
        mp.spawn(training_wrapper, args=(experiment, config_optimal),
                 nprocs=world_size, join=True)

    elif mode.startswith("test"):
        # e.g. test-linf, test-l1, test-l2, test-common
        torch.multiprocessing.set_start_method("spawn", force=True)

        test_type = mode.split("-")[1]  # "linf", "l1", "l2", "common"

        print(f"Testing step: {test_type}", flush=True)
        path = os.path.join(config_base.configs_path, "HPO_results",
                            config_base.project_name, f"{config_base.exp_id}.yaml")
        config_optimal = OmegaConf.load(path)

        os.makedirs(os.path.join(config_base.results_path, config_base.project_name),
                    exist_ok=True)
        experiment.launch_test(test_type, config_optimal)

    else:
        print(f"Unknown mode {mode}", flush=True)
        sys.exit(1)

if __name__ == "__main__":

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    
    main()
