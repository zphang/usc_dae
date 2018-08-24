def cpu(x):
    if isinstance(x, tuple):
        return tuple(cpu(_) for _ in x)
    return x.cpu() if x is not None else None


def gpu(x):
    if isinstance(x, tuple):
        return tuple(gpu(_) for _ in x)
    return x.cuda() if x is not None else None


default = cpu


def device_from_conf(conf):
    return device_from_env(conf.env)


def device_from_env(env):
    if env.use_gpu:
        return gpu
    else:
        return cpu
