from .env import check_env_flag

if check_env_flag('NO_MKLDNN'):
    WITH_MKLDNN = False
    MKLDNN_LIB_DIR = None
    MKLDNN_INCLUDE_DIR = None
else:
    WITH_MKLDNN = True
    MKLDNN_LIB_DIR = ''
    MKLDNN_INCLUDE_DIR = ''

#TODO: add avx512 compile flag to setup.py
if check_env_flag('WITH_AVX512'):
    WITH_AVX512 = True
else:
    WITH_AVX512 = False
