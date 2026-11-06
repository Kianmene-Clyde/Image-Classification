"""Liaison ctypes avec signatures explicites et état exportable sans pointeur."""
import ctypes as ct
import os
from pathlib import Path
import sys
import numpy as np
from .models.base import Classifier


def library_path():
    override = os.environ.get("SIGNLAB_NATIVE_LIBRARY")
    if override:
        return Path(override).resolve()
    suffix = "signlab_native.dll" if sys.platform == "win32" else "libsignlab_native.dylib" if sys.platform == "darwin" else "libsignlab_native.so"
    return Path(__file__).resolve().parents[1] / "build" / suffix


def load_library():
    path = library_path()
    if not path.is_file():
        raise ValueError(f"Bibliothèque native absente : {path}. Exécutez python scripts/build_native.py avec un compilateur C++17.")
    lib = ct.CDLL(str(path))
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    lib.sl_error.restype = ct.c_char_p
    lib.sl_error.argtypes = []
    lib.sl_create.argtypes = [ct.c_int] * 4 + [ct.c_double, ct.c_uint]
    lib.sl_create.restype = ct.c_void_p
    lib.sl_free.argtypes, lib.sl_free.restype = [ct.c_void_p], None
    lib.sl_fit.argtypes = [ct.c_void_p, doubles, ints, ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_uint]
    lib.sl_fit.restype = ct.c_int
    lib.sl_predict.argtypes, lib.sl_predict.restype = [ct.c_void_p, doubles, ct.c_int, doubles], ct.c_int
    lib.sl_state_size.argtypes, lib.sl_state_size.restype = [ct.c_void_p], ct.c_int
    lib.sl_export.argtypes, lib.sl_export.restype = [ct.c_void_p, doubles, ct.c_int], ct.c_int
    lib.sl_import.argtypes, lib.sl_import.restype = [doubles, ct.c_int], ct.c_void_p
    return lib


def check(lib, value):
    if value is None or value == -1:
        raise ValueError(lib.sl_error().decode("utf-8", errors="replace"))
    return value


class NativeClassifier(Classifier):
    def __init__(self, kind=0, hidden_units=24, epochs=250, learning_rate=0.1, l2=0.001, gamma=None, random_state=42):
        self.kind, self.hidden_units, self.epochs, self.learning_rate = kind, hidden_units, epochs, learning_rate
        self.l2, self.gamma, self.random_state = l2, gamma, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        X, y = np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.int32)
        lib = load_library()
        gamma = self.gamma if self.gamma is not None else 1 / X.shape[1]
        h = min(self.hidden_units, len(X))
        handle = check(lib, lib.sl_create(self.kind, X.shape[1], self.n_classes_, h, gamma, self.random_state))
        try:
            check(lib, lib.sl_fit(handle, X, y, len(X), self.epochs, self.learning_rate, self.l2, self.random_state))
            self.native_state_ = np.empty(check(lib, lib.sl_state_size(handle)), dtype=np.float64)
            check(lib, lib.sl_export(handle, self.native_state_, len(self.native_state_)))
        finally:
            lib.sl_free(handle)
        return self

    def decision_function(self, X):
        X = np.ascontiguousarray(self._check(X), dtype=np.float64)
        if (self.native_state_.ndim != 1 or len(self.native_state_) < 6
                or self.native_state_[2] != X.shape[1] or self.native_state_[3] != self.n_classes_):
            raise ValueError("Dimensions incompatibles avec l'état natif sauvegardé.")
        lib = load_library()
        handle = check(lib, lib.sl_import(self.native_state_, len(self.native_state_)))
        try:
            result = np.empty((len(X), self.n_classes_), dtype=np.float64)
            check(lib, lib.sl_predict(handle, X, len(X), result))
            return result
        finally:
            lib.sl_free(handle)
