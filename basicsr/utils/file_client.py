# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py  # noqa: E501
import numpy as np
from abc import ABCMeta, abstractmethod
import os.path as osp


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class Hdf5Backend(BaseStorageBackend):

    def __init__(self, h5_paths, client_keys='default', h5_clip='default', **kwargs):
        try:
            import h5py
        except ImportError:
            raise ImportError('Please install h5py to enable Hdf5Backend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(h5_paths, list):
            self.h5_paths = [str(v) for v in h5_paths]
        elif isinstance(h5_paths, str):
            self.h5_paths = [str(h5_paths)]
        assert len(client_keys) == len(self.h5_paths), ('client_keys and db_paths should have the same length, '
                                                        f'but received {len(client_keys)} and {len(self.h5_paths)}.')

        self._client = {}
        for client, path in zip(client_keys, self.h5_paths):
            try:
                self._client[client] = h5py.File(osp.join(path, h5_clip), 'r')
            except Exception:
                print(f"IO error, please check {path} {h5_clip}.")

    def get(self, filepath):

        file_lr = self._client['LR']
        file_hr = self._client['HR']
        img_lrs = []
        img_hrs = []

        # get images
        for idx in filepath:
            img_hr = file_hr[f'images/{idx:06d}'][:].astype(np.float32) / 255.
            img_hrs.append(img_hr)
        for idx in filepath:
            img_lr = file_lr[f'images/{idx:06d}'][:].astype(np.float32) / 255.
            img_lrs.append(img_lr)

        voxels_f = []
        voxels_b = []
        for idx in filepath[:-1]:
            voxel_f = file_lr[f'vFwd/{idx:06d}'][:].astype(np.float32)
            voxel_b = file_lr[f'vBwd/{idx:06d}'][:].astype(np.float32)
            voxels_f.append(voxel_f)
            voxels_b.append(voxel_b)
        voxels_Expo = []
        for idx in filepath:
            voxel_Expo = file_lr[f'vExpo/{idx:06d}'][:].astype(np.float32)
            voxels_Expo.append(voxel_Expo)
        assert len(voxels_Expo) == len(voxels_b) + 1, "Please check data."
        return img_lrs, img_hrs, voxels_f, voxels_b, voxels_Expo

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'hdf5': Hdf5Backend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)
