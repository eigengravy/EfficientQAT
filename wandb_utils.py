import os


def try_import_wandb():
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def init_wandb(args, phase, logger=None):
    wandb = try_import_wandb()
    if wandb is None:
        if logger:
            logger.info("wandb not installed, skipping logging")
        return None

    wandb_project = getattr(args, 'wandb_project', None)
    if not wandb_project:
        if logger:
            logger.info("--wandb_project not set, skipping wandb logging")
        return None

    scheme = getattr(args, 'scheme', 'baseline')
    net = getattr(args, 'net', None) or getattr(args, 'model_family', 'model')
    wbits = getattr(args, 'wbits', 4)
    group_size = getattr(args, 'group_size', 128)
    dataset = getattr(args, 'dataset', None) or getattr(args, 'calib_dataset', 'unknown')
    quant_config = f"w{wbits}g{group_size}"

    # Resume existing run if --wandb_run_id was passed (from train.py)
    run_id = getattr(args, 'wandb_run_id', None)
    if run_id:
        run = wandb.init(
            project=wandb_project,
            id=run_id,
            resume="must",
        )
    else:
        run_name = getattr(args, 'wandb_run_name', None)
        if not run_name:
            run_name = f"{scheme}-{net}-{quant_config}"

        tags = [scheme, net, quant_config, dataset]
        config = {k: v for k, v in vars(args).items()
                  if not k.startswith('_') and isinstance(v, (int, float, str, bool, type(None)))}
        config.update({
            "scheme": scheme,
            "model_family": net,
            "wbits": wbits,
            "group_size": group_size,
            "dataset": dataset,
            "quant_config": quant_config,
        })

        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config=config,
            tags=tags,
            reinit=True,
        )

    if logger:
        logger.info(f"wandb initialized: project={wandb_project}, run={run.name}, id={run.id}")
    return run


def finish_wandb(run):
    if run is not None:
        run.finish()
