'''
Functions:
    - hm_summary
        > Get information from the model
    - compute_n_mac
        > Complete model data with number of MAC
    - get_profile
        > Update the dictionary with MAC_profile
    - print_dict
        > Print the model data with MAC
    - compare_profile
        > Compare the profiled dict
    - print_comparison
        > Print the comparison
'''

###################
# Dependancy
###################
# Global packages
import torch
from collections import OrderedDict
import copy
from prettytable import PrettyTable

# Local packages
from mac_profile import MacProfile


#################################################
# MODIFIED SUMMARY
#################################################
'''
This function is a modified version of summary function:
    https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
'''
def hm_summary(model : torch.nn.Module, input_size : tuple, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    '''
    Input:
        > model: torch model"
        > input_size: (channels, height, width)
        > other by default
    Output:
        > summary: {layer: {input_size: [-1, c, h, w], output_size: [-1, c, h, w], trainable: bool, nb_params: int}
        (to get a specific data: summary[layer]["input_shape"])
    Additional function (return [[b0, c0, h0, w0], ..., [bn, cn, hn, wn]]):
        > input_shape = get_input_shape(summary_dict)
        > output_shape = get_output_shape(summary_dict)
    '''
    # Internal functions working on the local variable
    # REGISTER_HOOK
    ###############
    def hm_register_hook(module):

        
        # HOOK (summary dictionnary creation)
        ######
        def hm_hook(module, input, output):
            # Extract the class of the layer
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            # Extract the current index
            module_idx = len(summary)

            # Format Class-Idx (ex: Conv2d-1)
            m_key = "%s-%i" % (class_name, module_idx + 1)

            # Each layer is a dictionnary (input, output & parameters)
            summary[m_key] = OrderedDict()

            # Extract the input size (4D tensor)
            summary[m_key]["input_shape"] = list(input[0].size())
            # Define the  first index with the batch size
            summary[m_key]["input_shape"][0] = batch_size

            # Extract the output size (4D tensor)
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [batch_size] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            # Extract the number of parameters
            params = 0 # Reset params
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = int(params)
            
            # Extract the kernel size
            if hasattr(module, "kernel_size"):
                summary[m_key]["kernel_size"] = module.kernel_size
        ##########
        # END hook

        # Call the hook function (if there is any remaining layer)
        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hm_hook))

    ###################
    # END register_hook

    # Init dtype with the input_size
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Batch_size of 2 for batchnorm
    # Create the input tensor for the forward pass
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(hm_register_hook)

    # Make a forward pass
    # print(x.shape)
    model(*x)

    # Remove these hooks
    for h in hooks:
        h.remove()

    # Return summary
    return summary


#################################################
# COMPUTE_N_MAC
#################################################
def compute_n_mac(summary_dict):
    '''
    Input:
        > summary_dict: dictionary with model data
    Output:
        > summary_dict: updated dictionary with "N_MAC" (on convolution and linear layers)
        (to get a specific data: summary[layer]["input_shape"])
    RQ:
        The MAC are computed only on convolution and linear layers. 
        The others layers are not exectued on the MAC (may be executed on the host processor).
    '''
    # Incremental
    total_param = 0
    total_MAC = 0

    # Overwritten
    N_MAC = 0
    
    # Go throug the layers
    for layer in summary_dict:
        
        # If layer is either Conv or Linear (others are performed in host processor)
        if layer.startswith("Conv") or layer.startswith("Linear"):
            # Compute N_MAC
            if "kernel_size" in summary_dict[layer]:
                # MAC = (fc * fh * fw) * (mc * mh * mw) = (nc * fh * fw) * (mc * mh * mw)
                N_MAC = (summary_dict[layer]["input_shape"][1] * summary_dict[layer]["kernel_size"][0] * summary_dict[layer]["kernel_size"][1]) \
                * (summary_dict[layer]["output_shape"][1] * summary_dict[layer]["output_shape"][2] * summary_dict[layer]["output_shape"][3])
            else: # MAC = nc * mc
                N_MAC = summary_dict[layer]["input_shape"][1] * summary_dict[layer]["output_shape"][1]
            
            # Add N_MAC to the dictionnaries
            summary_dict[layer]["N_MAC"] = N_MAC
        # End if
    # End for
    
    # Return modified dictionaries with N_MAC
    return summary_dict

#################################################
# GET_PROFILE()
#################################################
def get_profile(mac_profile: MacProfile, summary_dict):
    '''
    Input:
        > mac_profile: gather the number of operations and number of memory access performed per MAC
        > summary_dict: the dictionary updated with N_MAC (function: compute_n_mac)
    Output:
        > profiled_dict: summary_dict updated with MAC_profile
    '''
    # Deepcopy summary_dict to unbound the data
    profiled_dict = copy.deepcopy(summary_dict)

    # Incremental data
    index = 0
    total_param = 0
    total_MAC = 0
    total_operations = 0
    total_memory_access = 0
    
    # Go in the layer
    for layer in summary_dict:
        # Overwritten data
        idx = ""
        N_param = "N/A"
        N_MAC = "N/A"
        N_operations = "N/A"
        N_memory_access = "N/A"

        # If layer is either Conv or Linear
        if layer.startswith("Conv") or layer.startswith("Linear"):
            # Increment the MAC index
            index = index + 1
            idx = index
            
            # Get N_param and N_MAC
            N_param = summary_dict[layer]["nb_params"]
            N_MAC = summary_dict[layer]["N_MAC"]

            # Compute N_operations and N_memory_access
            N_operations = int(N_MAC * mac_profile.N_operations)
            N_memory_access = N_MAC * mac_profile.N_memory_access

            # Increment the totals
            total_param = total_param + N_param
            total_MAC = total_MAC + N_MAC
            total_operations = total_operations + N_operations
            total_memory_access = total_memory_access + N_memory_access

        # End if

        # Update the dictionary
        profiled_dict[layer]["MAC_index"] = idx
        profiled_dict[layer]["N_MAC"] = N_MAC
        profiled_dict[layer]["N_operations"] = N_operations
        profiled_dict[layer]["N_memory_access"] = N_memory_access
    # End for
        
    # Create the TOTAL element to the dictionary
    profiled_dict["TOTAL"] = OrderedDict()

    # Add total
    profiled_dict["TOTAL"]["nb_params"] = total_param
    profiled_dict["TOTAL"]["N_MAC"] = total_MAC
    profiled_dict["TOTAL"]["N_operations"] = total_operations
    profiled_dict["TOTAL"]["N_memory_access"] = total_memory_access
    
    # Print table
    return profiled_dict

#################################################
# PRINT_DICT
#################################################
def print_dict(profiled_dict):
    '''
    Input:
        > profiled_dict: the dictionary updated with MAC_profile (function: get_profile)
    Output:
        => print data dictionnary for the given MAC profile
    '''
    # Create the table
    table = PrettyTable(["Layer", "MAC index", "Input Shape", "Output shape", "N_param", "N_MAC", "N_operations", "N_memory_access"])
    
    # Go in the layer
    for layer in profiled_dict:
        # Check the layer is not TOTAL
        if layer != "TOTAL":
            # Increment table
            table.add_row([layer, profiled_dict[layer]["MAC_index"], profiled_dict[layer]["input_shape"], profiled_dict[layer]["output_shape"], \
                        profiled_dict[layer]["nb_params"], profiled_dict[layer]["N_MAC"], profiled_dict[layer]["N_operations"], profiled_dict[layer]["N_memory_access"]])
        # End if
    # End for
    
    # Add total
    table.add_row(["TOTAL", "", "", "", "{0:,}".format(profiled_dict["TOTAL"]["nb_params"]), "{0:,}".format(profiled_dict["TOTAL"]["N_MAC"]), \
                    "{0:,}".format(profiled_dict["TOTAL"]["N_operations"]), "{0:,}".format(profiled_dict["TOTAL"]["N_memory_access"])])
    
    # Print table
    print(table)

#################################################
# COMPARE_PROFILE
#################################################
def compare_profile(nominal_profiled_dict, our_profiled_dict, oqa_profiled_dict):
    '''
    Input:
        > *_profiled_dict: profiled summary dictionaries (with function: get_profile)
    Output:
        > comparison_summary: dictionary
    '''
    # Create the output dictionary
    comparison_summary = OrderedDict()

    # Iterate over the layers
    for layer in nominal_profiled_dict:
        # If layer is either Conv or Linear
        if layer.startswith("Conv") or layer.startswith("Linear") or layer == "TOTAL":
            # Add a layer to comparison
            comparison_summary[layer] = OrderedDict()

            # Add the number of MAC
            comparison_summary[layer]["N_MAC"] = nominal_profiled_dict[layer]["N_MAC"]
            # Add the number of nominal operations (reference)
            comparison_summary[layer]["N_operations"] = nominal_profiled_dict[layer]["N_operations"]
            # Add the number of nominal memory access (reference)
            comparison_summary[layer]["N_memory_access"] = nominal_profiled_dict[layer]["N_memory_access"]

            # Compare Ours to nominal
            comparison_summary[layer]["ope_ours"] = our_profiled_dict[layer]["N_operations"]
            comparison_summary[layer]["ops_ours-nom"] = our_profiled_dict[layer]["N_operations"] - nominal_profiled_dict[layer]["N_operations"]
            comparison_summary[layer]["mem_ours-nom"] = our_profiled_dict[layer]["N_memory_access"] - nominal_profiled_dict[layer]["N_memory_access"]
            
            # Compare OQA to nominal
            comparison_summary[layer]["ope_oqa"] = oqa_profiled_dict[layer]["N_operations"]
            comparison_summary[layer]["ops_oqa-nom"] = oqa_profiled_dict[layer]["N_operations"] - nominal_profiled_dict[layer]["N_operations"]
            comparison_summary[layer]["mem_oqa-nom"] = oqa_profiled_dict[layer]["N_memory_access"] - nominal_profiled_dict[layer]["N_memory_access"]
        # End if
    # Enf for

    # Return the dictionary
    return comparison_summary

#################################################
# PRINT_COMPARISON
#################################################
def print_comparison(comparison_summary):
    '''
    Input:
        > comparison_summary: comparison summary dictionary (with function: compare_profile)
    Output:
        => print the dictionary
    '''
    # Create the table
    table = PrettyTable(["Layer", "Nominal MAC", "Nominal Operation", \
                         "Ours ope", "OQA ope", \
                         "Ope: Ours - nominal", "Ope: OQA - nominal", \
                         "Nominal memory access", \
                         "Memory: Ours - nominal", "Memory: OQA - nominal"])
    
    # Go in the layer
    for layer in comparison_summary:
        table.add_row([layer, "{0:,}".format(comparison_summary[layer]["N_MAC"]), "{0:,}".format(comparison_summary[layer]["N_operations"]), \
                       "{0:,}".format(comparison_summary[layer]["ope_ours"]), "{0:,}".format(comparison_summary[layer]["ope_oqa"]), \
                       "{0:,}".format(comparison_summary[layer]["ops_ours-nom"]), "{0:,}".format(comparison_summary[layer]["ops_oqa-nom"]), \
                       "{0:,}".format(comparison_summary[layer]["N_memory_access"]), \
                       "{0:,}".format(comparison_summary[layer]["mem_ours-nom"]), "{0:,}".format(comparison_summary[layer]["mem_oqa-nom"])])
    # End for

    # Print table
    print(table)