import os
from yacs.config import CfgNode
from .data_utils import InputExample, InputFeatures
from .data_sampler import FewShotSampler
from .data_processor import DataProcessor
from .lama_dataset import LAMAProcessor
from .relation_classification_dataset import TACREDProcessor, TACREVProcessor, ReTACREDProcessor, SemEvalProcessor
from .superglue_dataset import WicProcessor, RteProcessor, CbProcessor, WscProcessor, BoolQProcessor, CopaProcessor, MultiRcProcessor, RecordProcessor
from .typing_dataset import FewNERDProcessor
from .text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor
from .conditional_generation_dataset import WebNLGProcessor

from .typing_dataset import PROCESSORS as TYPING_PROCESSORS
from .text_classification_dataset import PROCESSORS as TC_PROCESSORS
from .superglue_dataset import PROCESSORS as SUPERGLUE_PROCESSORS
from .relation_classification_dataset import PROCESSORS as RC_PROCESSORS
from .lama_dataset import PROCESSORS as LAMA_PROCESSORS
from .conditional_generation_dataset import PROCESSORS as CG_PROCESSORS
from .lmbff_dataset import PROCESSORS as LMBFF_PROCESSORS

from openprompt.utils.logging import logger


PROCESSORS = {
    **TYPING_PROCESSORS,
    **TC_PROCESSORS,
    **SUPERGLUE_PROCESSORS,
    **RC_PROCESSORS,
    **LAMA_PROCESSORS,
    **CG_PROCESSORS,
    **LAMA_PROCESSORS,
    **LMBFF_PROCESSORS,
}


def load_dataset(config: CfgNode, return_class=True):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.
    
    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset
    processor = PROCESSORS[dataset_config.name.lower()]()
    try:
        if not config.dataset.domains:
            train_dataset = processor.get_train_examples(dataset_config.path)
        else:
            train_dataset = []
            if not config.dataset.adv_class and not config.dataset.kl_div:
                for domain in config.dataset.domains:
                    if domain != config.dataset.target_domain:
                        train_dataset.append(processor.get_train_examples(os.path.join(dataset_config.path, domain)))
                min_ins = len(train_dataset[0])
                for set in train_dataset:
                    if len(set) < min_ins:
                        min_ins = len(set)
                train_dataset = [set[:min_ins] for set in train_dataset]
            else:
                ent = []
                neu = []
                con = []
                all_min = 100000
                for domain in config.dataset.domains:
                    if domain != config.dataset.target_domain:
                        if config.dataset.name == 'amazonmulti':
                            train_dataset.append(processor.get_train_pos_examples(os.path.join(dataset_config.path, domain)))
                            train_dataset.append(processor.get_train_neg_examples(os.path.join(dataset_config.path, domain)))
                        elif config.dataset.name == 'mnlimulti':
                            ent.append(processor.get_train_ent_examples(os.path.join(dataset_config.path, domain)))
                            neu.append(processor.get_train_neu_examples(os.path.join(dataset_config.path, domain)))
                            con.append(processor.get_train_con_examples(os.path.join(dataset_config.path, domain)))
                            all_min = min(all_min, len(ent[-1]), len(neu[-1]), len(con[-1]))
                if config.dataset.name == 'mnlimulti':
                    for ent_d, neu_d, con_d in zip(ent, neu, con):
                        train_dataset.append(ent_d[:all_min])
                        train_dataset.append(neu_d[:all_min])
                        train_dataset.append(con_d[:all_min])


    except FileNotFoundError:
        logger.warning("Has no training dataset.")
        train_dataset = None
    try:
        if not config.dataset.domains:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        else:
            valid_dataset = processor.get_dev_examples(os.path.join(dataset_config.path, config.dataset.target_domain))
    except FileNotFoundError:
        logger.warning("Has no valid dataset.")
        valid_dataset = None
    try:
        if not config.dataset.domains:
            test_dataset = processor.get_test_examples(dataset_config.path)
        else:
            test_dataset = processor.get_test_examples(os.path.join(dataset_config.path, config.dataset.target_domain))
    except FileNotFoundError:
        logger.warning("Has no test dataset.")
        test_dataset = None
    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset