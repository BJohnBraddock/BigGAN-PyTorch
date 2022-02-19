import torch
import torch.nn as nn

import utils
import train_fns
from sync_batchnorm import patch_replication_callback
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from simple_utils import load_checkpoint

def run(config):

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]

    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)

    device = 'cuda'
    print("Using device cuda" if torch.cuda.is_available() else "Cuda device not found...")


    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name'] else utils.name_from_config(config))
    print('Experiment name is {}'.format(experiment_name))

    # Build the model
    G = model.Generator(**config).to(device)
    D = None
    G_ema, ema = None, None
    
    VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
    VCA = load_checkpoint(VCA, config['vca_filepath'])
    VCA = VCA.to(device)

    

    if config['G_fp16']:
        print('Casting G to float16')
        G = G.half()
           

    print(G)
    print('Number of params in G: {} '.format(sum([p.data.nelement() for p in G.parameters()])))


    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {
        'itr': 0,
        'epoch': 0,
        'save_num': 0,
        'save_best_num': 0,
        'best_IS': 0,
        'best_FID': 999999,
        'config': config
    }


    if config['resume']:
        print('Loading weights')
        utils.load_weights(G, D, state_dict,
                            config['load_weights_root'], experiment_name,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)
    

    

    

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    # test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
    #                                         experiment_name)
    # train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    # print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    # test_log = utils.MetricsLogger(test_metrics_fname, 
    #                                 reinitialize=(not config['resume']))
    # print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    # train_log = utils.MyLogger(train_metrics_fname, 
    #                             reinitialize=(not config['resume']),
    #                             logstyle=config['logstyle'])
    # # Write metadata
    # utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    


    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])

    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], device=device, fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()


    train = train_fns.VCA_generator_training_function(G, VCA, z_, y_, config)

    print(state_dict)

    # TODO: Training loop
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print(f"Epoch: {epoch}")

        for i in range(config['iters_per_epoch']):
            state_dict['itr'] += 1

            G.train()
            metrics = train()

            if not(state_dict['itr'] % config['log_every']):
                print('Epoch: {}    Itr: {}    G_loss: {}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))

        
        if config['G_eval_mode']:
            G.eval()
        print("Saving")
        print('Epoch: {}    Itr: {}    G_loss: {:e.4}    VCA_G_z: {}'.format(state_dict['epoch'], state_dict['itr'], metrics['G_loss'], metrics['VCA_G_z']))
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                            state_dict, config, experiment_name)
            
        
        state_dict['epoch'] += 1

    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name)
            

def main():

    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()