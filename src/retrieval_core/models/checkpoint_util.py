import os
import warnings

class CheckpointUtils:
    def __init__(self, checkpoint_dir):
        """
        Loads trained models from disk.
        :param checkpoint_dir: Checkpoint directory.
        """
        super().__init__()

        self._ckpt_dir = checkpoint_dir

    def _check_exists(self):
        if not os.path.exists(self._ckpt_dir):
            raise IOError('Checkpoint directory does not exist '
                          'and check points not found for testing the model!')

    def get_latest_ckpt_path(self):
        """
        Gets latest checkpoint path from directory
        :return:
        """
        def getint(name):
            basename, _, _ = name.partition('.')
            base, num = basename.split('_')
            return base, int(num)

        self._check_exists()
        ckpt_list = list([])
        base = 'epoch'
        for file in os.listdir(self._ckpt_dir):
            if file.endswith(".pth"):
                base, num = getint(file)
                ckpt_list.append(num)

        if len(ckpt_list) == 0:
            warnings.warn('No checkpoint found to resume state!', UserWarning)
            return None
        else:
            ckpt_name = base + '_' + str(sorted(ckpt_list)[-1]) + '.pth'
            print('Loading checkpoint: {}'.format(ckpt_name))
            return os.path.join(self._ckpt_dir, ckpt_name)

    def get_specfic_ckpt_path(self, name):
        """
        Gets specific checkpoint path from directory
        :param name: name of the checkpoint
        :return:
        """
        self._check_exists()
        return os.path.join(self._ckpt_dir, name + '.pth')