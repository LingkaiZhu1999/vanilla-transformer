config = {
    'name': 'de_en_translation',
    'embed_dim': 512,
    'n_blocks': 6,
    'n_heads': 8,
    'ff_hid_dim': 2048,
    'dropout': 0.1,
    'max_length': 100,
    'device': 'cuda',
    'lr': 0.0001,
    'clip': 1,
    'log_dir': 'logs_base',
    'weights_dir': 'weights_base',
    'save_interval': 50,
    'train_batch_size': 64,
    'val_batch_size': 128,
    'epochs': 6000,
    'warmup': 4000
}