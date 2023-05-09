import subprocess

import json

LOSSES = ['CPNLL', 'PNLL']
MODELS = ['AR', 'ARNet', 'LSTM', 'TGCN', 'GRU', 'ATGCN']

BASE_ARGS = [
    "--train_start", "2018-12-15",
    "--train_end", "2019-01-01",
    "--val_end", "2019-02-01",
    "--test_end", "2019-03-01",
    "--covariates",
    "--batch_size", "32",
    "--max_epochs", "1",
    "--censor_dynamic",
    "--loss", "CPNLL",
    "--sequence_length", "2",
    "--forecast_lead", "1",
    "--censor_level", "2",
    "--logger", "False",
    "--enable_progress_bar", "False",
    "--enable_model_summary", "False",
]

def run_main_py(parameters) -> bool:
    command = ['python', 'main.py']
    command.extend(parameters)
    result = subprocess.run(command, capture_output=True, text=True)
    index = parameters.index('--model_name') + 1
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print(f"{parameters[index]}.{parameters[index+2]}: ❌")
        return False
    else:
        print(f"{parameters[index]}.{parameters[index+2]}: ✅")
        return True

def setup_args(model, loss):
    args = BASE_ARGS.copy()
    args.extend(['--model_name', model])
    args.extend(['--loss', loss])

    # Only use censored mode for PNLL
    if 'CPNLL' in loss:
        args.extend(['--censored'])

    if 'TGCN' in model:
        args.extend(['--dataloader', 'EVChargersDatasetSpatial'])
    else:
        args.extend(['--dataloader', 'EVChargersDataset'])
        args.extend(['--cluster', "WEBSTER"])

    # if model is AR, remove hidden dim
    if model == 'AR' or model == 'ARNet':
        args.remove('--covariates')
    else:
        args.extend(['--hidden_dim', '16'])
    return args

if __name__ == '__main__':
    runs = []    
    for model in MODELS:
        for loss in LOSSES:
            # Get the arguments for this model and loss
            args = setup_args(model, loss)
            # Run the model
            runs.append(run_main_py(args))

    if all(runs):
        exit(0)
    else:
        exit(1)