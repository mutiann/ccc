from tensorflow.contrib.training import HParams

hparams = HParams(
    expr='atomic',

    train_batch_size=96,
    eval_batch_size=256,
    max_seq_length=48,  # 128 * 48 for 12GB GPU, 96 * 48 for 8GB; 32 for edge completion
    learning_rate=5e-5,
    lr_decay_step=10000,
    max_lr_decay_rate=0.1,
    adam_epsilon=1e-8,
    gradient_accumulation_steps=1,
    warmup_steps=1000,
    weight_decay=0.0,
    max_grad_norm=1.0,
    dropout_rate=0.3,

    use_type_classifier=False,
    n_edge_types=3,
    load_pretrained=True,
    use_roberta=False,

    sampling_rate=0.5,
    conceptualize_rate=0.5,  # among 1-sampling_rate
    entity_sub_mode='conceptualize',  # conceptualize / random_entity
    concept_score='frequency',  # likelihood / pmi / frequency
    score_weighted=True,
    random_select_mode='instance',  # entity / concept / instance
    n_candidates=40,  # small for conceptualize, large for random_entity
    text_sample_eval=True,
    text_sample_train=False,
    ht_symmetry=False,
    max_entity_word_inc=2,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
