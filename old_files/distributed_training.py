
    



# def convert_syncbn_to_bn(module):
#     module_output = module
#     if isinstance(module, torch.nn.SyncBatchNorm):
#         module_output = torch.nn.BatchNorm2d(
#             module.num_features,
#             eps=module.eps,
#             momentum=module.momentum,
#             affine=module.affine,
#             track_running_stats=module.track_running_stats,
#         )
#         if module.affine:
#             module_output.weight = module.weight
#             module_output.bias = module.bias
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#     else:
#         for name, child in module.named_children():
#             module_output.add_module(name, convert_syncbn_to_bn(child))
#     return module_output

                # print('got losses')
                #train_dataset.update_scores(rank, idxs, loss_values, logits)

                # List of tensors and their names for easy reference in the check
                # tensors = [loss_values, clean_values, robust_values, logits_nat, logits_adv]
                # tensor_names = ['loss_values', 'clean_values', 'robust_values', 'logits_nat', 'logits_adv']

                # check_for_nans(tensors, tensor_names)

                # # Check for negative values and raise an error if found
                # if (loss_values < 0).any():
                #     raise ValueError("The tensor 'loss_values' contains negative values.")
    
                # loss.backward()
                # optimizer.step()
                # print('backward')
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                # break


                # print('before update',rank, train_dataset.Sigma_inv, train_dataset.mu)
            # train_dataset.update_contextual_TS_parameters()
            # print('after update',rank, train_dataset.Sigma_inv, train_dataset.mu)

            # gradient_norm = compute_gradient_norms(model)
            # current_lr = optimizer.param_groups[0]['lr']
            # experiment.log_metric("iteration", iteration, epoch=iteration)
            # experiment.log_metric("loss_value", loss, epoch=iteration)
            # experiment.log_metric("clean_value", clean_values.mean(), epoch=iteration)
            # experiment.log_metric("adv_value", robust_values.mean(), epoch=iteration)
            # experiment.log_metric("lr_schedule", current_lr, epoch=iteration)
            # experiment.log_metric("gradient_norm", gradient_norm, epoch=iteration)
            #experiment.log_metric("reward", loss_values.sum(), epoch=iteration)  

            # if iteration % 5 == 0:
            #     print('start validation') 
            #     self.validate(valloader, model, experiment, iteration+1, rank)

            # if self.args.pruning_strategy in ['decay_based', 'decay_based_v2',  'decay_based_v3']:
            #     indices = train_sampler.process_indices
            #     train_dataset.decay_model.reset_counters()
            #     results = torch.tensor([ train_dataset.decay_model.fit_predict( train_dataset.global_scores2[idx], ) for idx in indices ])
            #     experiment.log_metric("solver_fails", train_dataset.decay_model.fail, epoch=iteration)
            #     experiment.log_metric("solver_total", train_dataset.decay_model.total, epoch=iteration)
            #     results = results.to(dtype=torch.float32)
            #     train_dataset.alphas[indices] = results[:,0]
            #     train_dataset.betas[indices] = results[:,1]
            #     train_dataset.cetas[indices] = results[:,2]
            #     train_dataset.pred_decay[indices] = results[:,4]
    
    # def final_validation(self, test_dataset, model, experiment, iteration, rank):
        
    #         # Re-instantiate the model
    #         model_eval = load_architecture(self.args)
    #         model_eval = CustomModel(self.args, model_eval)
    #         model_eval.set_fine_tuning_strategy('full_fine_tuning')
    #         model_eval.to(rank)

    #         model_eval.load_state_dict(model.module.state_dict())
    #         # Convert SyncBatchNorm to BatchNorm
    #         model_eval = convert_syncbn_to_bn(model_eval)
    #         model_eval.eval()
    #         model_eval = model_eval.to(rank)  # Ensure the model is on the correct device

    #         # Create DataLoader for the test_dataset
    #         b_size = 2#self.setup.test_batch_size()
    #         testloader = DataLoader(
    #             test_dataset, batch_size=b_size, shuffle=False, num_workers=0, pin_memory=False
    #         )

    #         print('start AA accuracy')
    #         total_correct_nat, total_correct_adv, total_examples = self.compute_AA_accuracy(
    #             testloader, model_eval, rank
    #         )
    #         print('end AA accuracy')

    #         # Compute the metrics
    #         clean_accuracy = total_correct_nat / total_examples
    #         robust_accuracy = total_correct_adv / total_examples

    #         # Log the metrics
    #         experiment.log_metric("final_clean_accuracy", clean_accuracy, epoch=iteration)
    #         experiment.log_metric("final_robust_accuracy", robust_accuracy, epoch=iteration)

    #         return clean_accuracy, robust_accuracy
    
    # def validate(self, valloader, model, experiment, iteration, rank):
    #     total_loss, total_correct_nat, total_correct_adv, total_examples = self.validation_metrics(valloader, model, rank)
    #     dist.barrier() 
    #     avg_loss, clean_accuracy, robust_accuracy  = self.sync_validation_results(total_loss, total_correct_nat, total_correct_adv, total_examples, rank)
    #     experiment.log_metric("val_loss", avg_loss, epoch=iteration)
    #     experiment.log_metric("val_clean_accuracy", clean_accuracy, epoch=iteration)
    #     experiment.log_metric("val_robust_accuracy", robust_accuracy, epoch=iteration)

    # def validation_metrics(self, valloader, model, rank):

    #     model.eval()

    #     total_loss = 0.0
    #     total_correct_nat = 0
    #     total_correct_adv = 0
    #     total_examples = 0

    #     for batch_id, batch in enumerate( valloader ):

    #             data, target, idxs = batch

    #             data, target = data.to(rank), target.to(rank) 

    #             loss_values, _, _, logits_nat, logits_adv = get_eval_loss(self.args, model, data, target, )

    #             total_loss += loss_values.sum().item()
    #             # Compute predictions
    #             preds_nat = torch.argmax(logits_nat, dim=1)  # Predicted classes for natural examples
    #             preds_adv = torch.argmax(logits_adv, dim=1)  # Predicted classes for adversarial examples

    #             # Accumulate correct predictions
    #             total_correct_nat += (preds_nat == target).sum().item()
    #             total_correct_adv += (preds_adv == target).sum().item()
    #             total_examples += target.size(0)
    #             break

    #     return total_loss, total_correct_nat, total_correct_adv, total_examples


# if self.args.lora:
#     add_lora(target_layers, model)
#     set_lora_gradients(args, model, target_layers)
    
# Do not use dist.barrier() here


    # def final_validation(self, test_dataset, model, experiment, iteration, rank ):

    #     model_eval = load_architecture(self.args).to(rank)
    #     model_eval.load_state_dict(model.module.state_dict())
    #     # Convert SyncBatchNorm to BatchNorm
    #     model_eval = convert_syncbn_to_bn(model_eval)

    #     model.eval()

    #     # Create a new testloader without DistributedSampler
    #     total_size = len(test_dataset)
    #     per_process_size = total_size // self.world_size

    #     # Calculate start and end indices for each process
    #     start_idx = rank * per_process_size
    #     # Ensure the last process gets any remaining data
    #     end_idx = (rank + 1) * per_process_size if rank != self.world_size - 1 else total_size

    #     # Subset the dataset
    #     subset_indices = list(range(start_idx, end_idx))
    #     subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    #     print(rank, len(subset_indices) )

    #     # Create DataLoader for the subset
    #     testloader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    #     print('start AA accuracy')
    #     total_correct_nat, total_correct_adv, total_examples = self.compute_AA_accuracy(testloader, model, rank)
    #     print('end AA accuracy')

    #     dist.barrier() 
    #     clean_accuracy, robust_accuracy  = self.sync_final_result(total_correct_nat, total_correct_adv, total_examples, rank)
        
        
    #     experiment.log_metric("final_clean_accuracy", clean_accuracy, epoch=iteration)
    #     experiment.log_metric("final_robust_accuracy", robust_accuracy, epoch=iteration)

    # def sync_final_result(self, total_correct_nat, total_correct_adv, total_examples, rank):

    #     # Aggregate results across all processes
    #     total_correct_nat_tensor = torch.tensor([total_correct_nat], dtype=torch.float32, device=rank)
    #     total_correct_adv_tensor = torch.tensor([total_correct_adv], dtype=torch.float32, device=rank)
    #     total_examples_tensor = torch.tensor([total_examples], dtype=torch.float32, device=rank)

    #     dist.all_reduce(total_correct_nat_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_correct_adv_tensor, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)

    #     # Compute global averages
    #     clean_accuracy = total_correct_nat_tensor.item() / total_examples_tensor.item()
    #     robust_accuracy = total_correct_adv_tensor.item() / total_examples_tensor.item()

    #     return clean_accuracy, robust_accuracy