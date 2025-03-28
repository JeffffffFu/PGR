from typing import Annotated
from baseline.GAP_master.core import console
from baseline.GAP_master.core import globals
from baseline.GAP_master.core.args.utils import ArgInfo
from baseline.GAP_master.core.loggers.base import LoggerBase
from baseline.GAP_master.core.loggers.csv import CSVLogger
from baseline.GAP_master.core.loggers.wandb import WandbLogger


class Logger:
    supported_loggers = {
        'csv': CSVLogger,
        'wandb': WandbLogger,
    }

    _instance: LoggerBase = None
    
    @classmethod
    def get_instance(cls) -> LoggerBase:
        return cls._instance

    def __new__(cls) -> LoggerBase:
        if cls._instance is None:
            raise RuntimeError('Logger is not initialized, call Logger.setup() first')
        return cls._instance

    @classmethod
    def setup(cls,
        logger:   Annotated[str,  ArgInfo(help='select logger type', choices=supported_loggers)] = 'csv',
        **kwargs: Annotated[dict, ArgInfo(help='additional kwargs for the underlying logger', bases=[LoggerBase])],
        ) -> LoggerBase:
        
        if globals['debug']:
            logger = 'wandb'
            kwargs['enabled'] = True
            kwargs['project'] += '-DEBUG'
            console.debug(f'debug mode: wandb logger is enabled for project {kwargs["project"]}')
        
        LoggerCls = cls.supported_loggers[logger]
        cls._instance = LoggerCls(**kwargs)
        return cls._instance
