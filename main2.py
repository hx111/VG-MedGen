import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import datetime
import argparse

from sklearn.model_selection import train_test_split

import utils
import models
import supervision

from loaders import ham_loader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_arguments(args):
    usage_text = (
        "SDNet Pytorch Implementation for HAM10000"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument('-e', '--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Number of inputs per batch')  # 建议bs可以稍大一些
    parser.add_argument('-n', '--name', type=str, default='sdnet_ham10000',
                        help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn', '--model_name', type=str, default='sdnet',
                        help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr', '--learning_rate', type=float, default='0.0001',
                        help='The learning rate for model training')
    parser.add_argument('-wi', '--weight_init', type=str, default="xavier",
                        help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    parser.add_argument("--anatomy_factors", type=int, default=8, help='Number of anatomy factors to encode')
    parser.add_argument("--modality_factors", type=int, default=8, help='Number of modality factors to encode')
    parser.add_argument("--charbonnier", type=int, default=0,
                        help='Choose Charbonnier penalty for the reconstruction loss')
    parser.add_argument("--data_path", type=str, default='../dataset/HAM10000',
                        help='Path to HAM10000 dataset root')
    parser.add_argument("--kl_w", type=float, default=0.01, help='KL divergence loss weight')
    parser.add_argument("--regress_w", type=float, default=1.0,
                        help='Regression loss weight (z-reconstruction)')
    parser.add_argument("--focal_w", type=float, default=0.0, help='Focal loss weight')
    parser.add_argument("--dice_w", type=float, default=10.0, help='Dice loss weight')
    parser.add_argument("--reco_w", type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument("--perceptual_w", type=float, default=0.1, help='Perceptual loss weight')

    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to use for dataload')
    parser.add_argument('-d', '--disp_iters', type=int, default=1,
                        help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom', type=str, nargs='?', default=None, const="127.0.0.1",
                        help="Visdom server IP (port defaults to 8097)")
    parser.add_argument('--visdom_iters', type=int, default=10,
                        help="Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument('--print_factors', type=int, default=0,
                        help='Set to 1 to visualize the anatomy factors in Visdom')

    return parser.parse_known_args(args)


if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    print('{} | Torch Version: {}'.format(datetime.datetime.now(), torch.__version__))
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device(
        'cuda:{}'.format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else 'cpu')
    print('Training {0} for {1} epochs using a batch size of {2} on {3}'.format(args.name, args.epochs, args.batch_size,
                                                                                device))

    torch.manual_seed(667)
    if device.type == 'cuda':
        torch.cuda.manual_seed(667)

    visualizer = utils.visualization.NullVisualizer() if args.visdom is None \
        else utils.visualization.VisdomVisualizer(args.name, args.visdom, count=1)

    model_params = {
        'width': 224,
        'height': 224,
        'ndf': 64,
        'norm': "batchnorm",
        'upsample': "nearest",
        'num_classes': 1,
        'anatomy_out_channels': args.anatomy_factors,
        'z_length': args.modality_factors,
        'num_mask_channels': 8,
    }
    model = models.get_model(args.model_name, model_params)
    num_params = utils.count_parameters(model)
    print('Model Parameters: ', num_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)

    print("Initializing HAM10000 loader...")
    loader = ham_loader.HAM10000Loader(args.data_path, input_shape=(model_params['width'], model_params['height']))

    full_train_data = loader.load_dataset(split_type='train')
    print(f"Full training data loaded: {full_train_data.images.shape[0]} samples")

    indices = list(range(full_train_data.images.shape[0]))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

    images = torch.from_numpy(full_train_data.images[train_indices]).float()
    masks = torch.from_numpy(full_train_data.masks[train_indices]).float()
    print(f"Internal training set created: {images.shape[0]} samples")

    vimages = torch.from_numpy(full_train_data.images[val_indices]).float()
    vmasks = torch.from_numpy(full_train_data.masks[val_indices]).float()
    print(f"Internal validation set created: {vimages.shape[0]} samples")

    l1_distance = nn.L1Loss().to(device)
    from supervision.losses import PerceptualLoss

    perceptual_loss_fn = PerceptualLoss().to(device)

    optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)

    total_loss = supervision.AverageMeter()
    running_reco_loss = supervision.AverageMeter()
    running_kl_loss = supervision.AverageMeter()
    running_dice_loss = supervision.AverageMeter()
    running_reg_loss = supervision.AverageMeter()
    running_focal_loss = supervision.AverageMeter()
    running_kl_a_loss = supervision.AverageMeter()
    running_reg_a_loss = supervision.AverageMeter()
    running_perceptual_loss = supervision.AverageMeter()

    val_running_dice_score = supervision.AverageMeter()

    b_images = torch.zeros(args.batch_size, 3, model_params['height'], model_params['width'])
    b_masks = torch.zeros(args.batch_size, model_params['num_classes'] + 1, model_params['height'],
                          model_params['width'])
    collapsed_b_masks = torch.zeros(args.batch_size, 1, model_params['height'], model_params['width'])
    v_image = torch.zeros(1, 3, model_params['height'], model_params['width'])
    v_mask = torch.zeros(1, model_params['num_classes'] + 1, model_params['height'],
                         model_params['width'])

    total_batches = images.shape[0] // args.batch_size
    global_iterations = 0
    val_dice_best = 0.0

    for epoch in range(args.epochs):
        idx = torch.randperm(images.shape[0])
        in_batch_iter = 0
        model.train()

        for iteration in range(images.shape[0]):
            if (iteration + args.batch_size) > images.shape[0]:
                break
            if in_batch_iter < args.batch_size:
                b_images[in_batch_iter] = images[idx[iteration]]

                cmask = masks[idx[iteration]]

                tmask_0 = 1.0 - cmask
                tmask = torch.cat([tmask_0, cmask], dim=0)

                b_masks[in_batch_iter] = tmask
                in_batch_iter += 1
            else:
                kl_loss = torch.tensor(0.0, device=device)
                dice_loss = torch.tensor(0.0, device=device)
                regression_loss = torch.tensor(0.0, device=device)
                focal_loss = torch.tensor(0.0, device=device)
                kl_a_loss = torch.tensor(0.0, device=device)
                regression_a_loss = torch.tensor(0.0, device=device)
                perceptual_loss = torch.tensor(0.0, device=device)
                optimizer.zero_grad()

                collapsed_b_masks = b_masks[:, 1, :, :].squeeze()

                reco, z_out, mu_tilde, a_mu_tilde, a_out, seg_pred, mu, logvar, a_mu, a_logvar = model(
                    b_images.to(device), b_masks.to(device), 'training')
                if args.charbonnier > 0:
                    l1_loss = l1_distance(reco, b_images.to(device))
                    reco_loss = supervision.charbonnier_penalty(l1_loss)
                else:
                    reco_loss = l1_distance(reco, b_images.to(device))

                if args.perceptual_w > 0.0:
                    perceptual_loss = perceptual_loss_fn(reco, b_images.to(device))
                if args.kl_w > 0.0:
                    kl_loss = supervision.KL_divergence(logvar, mu)
                if args.dice_w > 0.0:
                    dice_loss = supervision.dice_loss(seg_pred[:, 1:, :, :], b_masks[:, 1:, :, :].to(device))
                if args.regress_w > 0.0:
                    regression_loss = l1_distance(mu_tilde, z_out.detach())
                if args.focal_w > 0.0:
                    focal_loss = supervision.FocalLoss(gamma=2, alpha=0.25)(seg_pred, collapsed_b_masks.to(device))

                if args.model_name == 'sdnet2' and args.kl_w > 0.0:
                    kl_a_loss = supervision.KL_divergence(a_logvar, a_mu)
                if args.model_name == 'sdnet2' and args.regress_w > 0.0:
                    regression_a_loss = l1_distance(a_mu_tilde, a_out)
                if args.model_name == 'sdnet3' and args.kl_w > 0.0:
                    kl_a_loss = supervision.KL_divergence(a_logvar, a_mu)

                batch_loss = args.reco_w * reco_loss \
                             + args.kl_w * kl_loss \
                             + args.dice_w * dice_loss \
                             + args.regress_w * regression_loss \
                             + args.focal_w * focal_loss \
                             + args.kl_w * kl_a_loss \
                             + args.regress_w * regression_a_loss

                batch_loss += args.perceptual_w * perceptual_loss
                batch_loss.backward()
                optimizer.step()
                total_loss.update(batch_loss.detach())
                running_reco_loss.update(reco_loss.detach())
                running_kl_loss.update(kl_loss.detach())
                running_dice_loss.update(dice_loss.detach())
                running_reg_loss.update(regression_loss.detach())
                running_focal_loss.update(focal_loss.detach())
                running_kl_a_loss.update(kl_a_loss.detach())
                running_reg_a_loss.update(regression_a_loss.detach())
                running_perceptual_loss.update(perceptual_loss.detach())

                if (iteration + 1) % args.visdom_iters == 0 and args.visdom is not None:
                    print(f"Visualizing images at iteration {iteration + 1}...")
                    visualizer.show_map(b_images.to(device), 'Input Image')
                    visualizer.show_map(reco, 'Reconstructed Image')
                    visualizer.show_seg_map(b_masks.to(device), 'GT Mask')
                    visualizer.show_seg_map(seg_pred, 'Predicted Mask')
                    if args.print_factors:
                        visualizer.show_anatomical_factors(a_out, 'Anatomical Factor')

                if (iteration + 1) % args.disp_iters <= args.batch_size:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                    print(
                        "Epoch: {}, iteration: {}/{}\nLR: {}\nFocal: {}\nDice: {}\nReco: {}\nPerceptual: {}\nKLD: {}\nReg: {}\nKLD_a: {}\nReg_a: {}\nTotal average loss: {}\n\n" \
                            .format(epoch, iteration, images.shape[0], lr, running_focal_loss.avg,
                                    running_dice_loss.avg,
                                    running_reco_loss.avg, running_perceptual_loss.avg, running_kl_loss.avg,
                                    running_reg_loss.avg, running_kl_a_loss.avg,
                                    running_reg_a_loss.avg, total_loss.avg))

                if args.visdom is not None:
                    visualizer.append_loss(epoch, global_iterations, total_loss.avg.item(), "Total")
                    visualizer.append_loss(epoch, global_iterations, running_reco_loss.avg.item(), "Reconstruction")
                    visualizer.append_loss(epoch, global_iterations, running_focal_loss.avg.item(), "Focal")
                    visualizer.append_loss(epoch, global_iterations, running_kl_loss.avg.item(), "KLD")
                    visualizer.append_loss(epoch, global_iterations, running_dice_loss.avg.item(), "Dice")
                    visualizer.append_loss(epoch, global_iterations, running_reg_loss.avg.item(), "Regression")
                    visualizer.append_loss(epoch, global_iterations, running_kl_a_loss.avg.item(), "KLD_a")
                    visualizer.append_loss(epoch, global_iterations, running_reg_a_loss.avg.item(), "Regression_a")
                    visualizer.append_loss(epoch, global_iterations, running_perceptual_loss.avg.item(), "Perceptual")
                total_loss.reset()
                running_reco_loss.reset()
                running_kl_loss.reset()
                running_dice_loss.reset()
                running_reg_loss.reset()
                running_focal_loss.reset()
                running_kl_a_loss.reset()
                running_reg_a_loss.reset()
                running_perceptual_loss.reset()

                global_iterations += args.batch_size
                in_batch_iter = 0

        with torch.no_grad():
            model.eval()
            for iteration in range(vimages.shape[0]):
                v_image[0] = vimages[iteration]
                cmask = vmasks[iteration]
                tmask_0 = 1.0 - cmask
                tmask = torch.cat([tmask_0, cmask], dim=0)
                v_mask[0] = tmask

                _, _, _, _, _, seg_pred, _, _, _, _ = model(v_image.to(device), v_mask.to(device), 'val')

                dice_score_val = supervision.dice_score(seg_pred[:, 1:, :, :], v_mask[:, 1:, :, :].to(device))
                val_running_dice_score.update(dice_score_val)

            print("Epoch: {},\nValidation Samples: {}\nDice Score: {}\n" \
                  .format(epoch, iteration + 1, val_running_dice_score.avg))

            if args.visdom is not None:
                visualizer.append_loss(epoch, epoch, val_running_dice_score.avg.item(), "Validation Dice Score")

            val_dice_curr = val_running_dice_score.avg.item()
            scheduler.step(val_dice_curr)

            if val_dice_curr > val_dice_best:
                val_dice_best = val_dice_curr
                print("Epoch checkpoint: New best validation Dice score {:.4f}".format(val_dice_best))
                current_dir = os.getcwd()
                final_dir = os.path.join(current_dir, args.save_path)
                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)
                utils.save_network_state(model, model_params['width'], model_params['height'], model_params['ndf'], \
                                         model_params['norm'], model_params['upsample'], model_params['num_classes'], \
                                         model_params['anatomy_out_channels'], model_params['z_length'], \
                                         model_params['num_mask_channels'], optimizer, \
                                         epoch, args.name + "_model_state_epoch_" + str(epoch), \
                                         final_dir)
            val_running_dice_score.reset()
