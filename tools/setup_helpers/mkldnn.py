from .env import check_env_flag

if check_env_flag('NO_MKLDNN'):
    WITH_MKLDNN = False
else:
    WITH_MKLDNN = True

if check_env_flag('WITH_AVX512'):
    WITH_AVX512 = True
else:
    WITH_AVX512 = False
