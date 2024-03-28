##############
# Dependencies
##############
# Installed packages
import torch
import torchvision

# Local packages
from functions import *
from mac_profile import MacProfile

###############
# MAIN FUNCTION
###############
def main():
    # Define the mac_profile
    ########################
    # Nominal: memory = 2 read + 1 write, operations = 1 multiply + 1 accumulate, cycles = 2
    nominal_profile = MacProfile(N_memory_access=3, N_operations=2, N_cycles=2)
    # Ours: memory = 2 read + 1 write, operations = 2*(1 multiply + 1 accumulate), cycles = 3
    our_profile = MacProfile(N_memory_access=3, N_operations=4, N_cycles=3)
    # OQA: memory = 2 read + 1 write, operations = 1 multiply + 1 accumulate, cycles = 2
    oqa_profile = MacProfile(N_memory_access=3, N_operations=2, N_cycles=2)

    # User interactions
    ###################
    # Model selection
    user_model = input("Tap 'r' to use ResNet-18, tap 'v' to use VGG-16\n")
    # Print model details
    user_print_model = input("Do you want to print model? ('y' or 'n')\n")
    user_show_dict = input("Do you want to print model data dictionary? If yes, which MAC_profile: \n Nominal: \t '0' \n Ours: \t\t '1' \n OQA: \t\t '2' \n None: \t\t 'n'\n")


    # Model declaration
    ###################
    if (user_model == 'r'):
        model = torchvision.models.resnet18()
        model_name = "Resnet-18"
    elif (user_model == 'v'):
        model = torchvision.models.vgg16()
        model_name = "VGG-16"
    else:
        raise ValueError('\nUser input does not fit expected input (0 or 1)\n')

    # Print the model if required
    if (user_print_model == 'y'):
        print(f"Model {model_name}: \n{model} \n")
    else: # Do nothing
        pass
    
    # Get the data dictionary from the model
    ########################################
    # Get the summary
    model_dict = hm_summary(model=model, input_size=(3, 224, 224))

    # Compute the number of MAC
    updated_model_dict = compute_n_mac(model_dict)
    #print(f"THE UPDATED MODEL UPDATED: \n{updated_model_dict}\n")

    # Profile the dictionaries
    ##########################
    # Get the profile
    nominal_profiled_dict = get_profile(mac_profile=nominal_profile, summary_dict=updated_model_dict)
    our_profiled_dict = get_profile(mac_profile=our_profile, summary_dict=updated_model_dict)
    oqa_profiled_dict = get_profile(mac_profile=oqa_profile, summary_dict=updated_model_dict)
    
    # Print the dictionary if required:
    if (user_show_dict == '0'): # Nominal
        print(f"\nThe data dictionary with nominal MAC profile: \n\t \
              N_memory_access/MAC = {nominal_profile.N_memory_access} \t N_operations/MAC = {nominal_profile.N_operations} \t N_cycles/MAC = {nominal_profile.N_cycles}")
        print_dict(nominal_profiled_dict)
    elif (user_show_dict == '1'): # Ours
        print(f"\nThe data dictionary with our MAC profile: \n\t \
              N_memory_access/MAC = {our_profile.N_memory_access} \t N_operations/MAC = {our_profile.N_operations} \t N_cycles/MAC = {our_profile.N_cycles}")
        print_dict(our_profiled_dict)
    elif (user_show_dict == '2'): # OQA
        print(f"\nThe data dictionary with OQA MAC profile: \n\t \
              N_memory_access/MAC = {oqa_profile.N_memory_access} \t N_operations/MAC = {oqa_profile.N_operations} \t N_cycles/MAC = {oqa_profile.N_cycles}")
        print_dict(oqa_profiled_dict)
    elif (user_show_dict == 'n'): # Do nothing
        pass
    else: # Unexpected input
        raise ValueError('\nUser input does not fit expected input\n')
    
    # Compare the dictionaries
    ##########################
    # Compare the dictionaries
    comparison_summary = compare_profile(nominal_profiled_dict, our_profiled_dict, oqa_profiled_dict)

    # Print the comparison
    print(f"\n\nCOMPARISON SUMMARY for {model_name}: \n")
    print_comparison(comparison_summary)

###########
# Execution
###########

if __name__ == '__main__':
    main()
