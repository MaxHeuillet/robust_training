# import torch


def uncertainty_sampling(rank, args):
    state_dict, pool_dataset, N, top_n_indices = args
    setup(self.world_size, rank)
    sampler = DistributedSampler(pool_dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
    loader = DataLoader(pool_dataset, batch_size=self.batch_size_uncertainty, sampler=sampler, num_workers=self.world_size, shuffle=False)
    model = self.load_model()
    model.load_state_dict(state_dict)
    model.to(rank)
    model.eval()  # Ensure the model is in evaluation mode
    model = DDP(model, device_ids=[rank])
    print('do the predictions')
    predictions = []
    indices_list = []
    with torch.no_grad():
        for inputs, _, indices in loader:
            inputs = inputs.cuda(rank)
            outputs = model(inputs) 
            predictions.append(outputs)
            indices_list.append( indices.cuda(rank) )
    print('prepare the predictions')
    predictions = torch.cat(predictions, dim=0)
    outputs = torch.softmax(predictions, dim=1)
    uncertainty = 1 - torch.max(outputs, dim=1)[0]
    gather_list = [torch.zeros_like(uncertainty) for _ in range(self.world_size)]
    
    indices = torch.cat(indices_list, dim=0)
    index_gather_list = [torch.zeros_like(indices) for _ in range(self.world_size)]
    print('gather the predictions')
    dist.all_gather(gather_list, uncertainty)
    dist.all_gather(index_gather_list, indices)
    dist.barrier()  # Synchronization point
    if rank == 0:
        all_uncertainties = torch.cat(gather_list, dim=0)
        all_indices = torch.cat(index_gather_list, dim=0)
        _, top_n_indices_sub = torch.topk(all_uncertainties, N, largest=True)
        top_n_indices.copy_(all_indices[top_n_indices_sub])
    cleanup()