import src.runners.dae_train as dae_train
import src.utils.conf as conf
if __name__ == "__main__":
    config_ = conf.Configuration.from_json_arg()
    trainer = dae_train.DAETrainer.from_config(config_)
    trainer.run_train()
