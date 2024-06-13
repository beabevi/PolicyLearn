from setuptools import setup
from torch.utils import cpp_extension

setup(
    author="Beatrice Bevilacqua",
    name="policy_learn",
    ext_modules=[
        cpp_extension.CppExtension(
            "sparse_utils",
            ["csrc/vrange.cpp", "csrc/vrange_kernel.cu"],
            library_dirs=["/u/ml00_s/bbevilac/miniconda3/envs/policy-learn/lib"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    install_requires=["filelock==3.10.7", "hydra-core==1.3.2"],
)
