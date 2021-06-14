import torch


def L2SP(model, w0_dic, new_layers, num_lowlrs):
    existing_l2_reg = None
    new_l2_reg = None

    for name, w in model.named_parameters():
        # print(name)
        # if 'weight' not in name:  # I don't know if that is true: I was told that Facebook regularized biases too.
        #    continue
        # if 'downsample.1' in name:  # another bias
        #    continue
        if 'bn' in name:  # bn parameters
            continue
        # fc layers
        if 'linear' in name:
            if new_l2_reg is None:
                new_l2_reg = torch.pow(w, 2).sum() / 2
            else:
                new_l2_reg += torch.pow(w, 2).sum() / 2
        # new layers must be l2 not L2SP as there is no w0
        elif (int(name.split('.')[1]) >= (22 - new_layers)):
            if new_l2_reg is None:
                new_l2_reg = torch.pow(w, 2).sum() / 2
            else:
                new_l2_reg += torch.pow(w, 2).sum() / 2
            # print('new: ' + str(new_l2_reg) + ' ' + str(name))
        else:
            # print(name)
            w0 = w0_dic[name].data
            w0 = w0.cuda()
            # split layers the same way as high and low learning rates so we can do different things with them
            # note - the number of weights in the fc layer is orders of magnitude great than those in the conv layer
            # also lower layers should be more transferable so keeping them closer to their pretrained values is more likely to be optimal
            if (int(name.split('.')[1]) < (num_lowlrs)):
                if existing_l2_reg is None:
                    existing_l2_reg = 10 * (torch.pow(w - w0, 2).sum() / 2)
                else:
                    existing_l2_reg += 5 * (torch.pow(w - w0, 2).sum() / 2)
                # print('existing: ' + str(existing_l2_reg) + ' ' + str(name))
            else:
                if new_l2_reg is None:
                    new_l2_reg = torch.pow(w, 2).sum() / 2
                else:
                    new_l2_reg += torch.pow(w, 2).sum() / 2
                # if existing_l2_reg is None:
                #    existing_l2_reg = (torch.pow(w-w0, 2).sum()/2)
                # else:
                #    existing_l2_reg += (torch.pow(w-w0, 2).sum()/2)

    # print('existing: ' + str(existing_l2_reg))
    # print('new: ' + str(new_l2_reg))
    l2_reg = existing_l2_reg * 0.004 + new_l2_reg * 0.0005
    # print(l2_reg)
    return l2_reg
