from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fused_graph_ops',
    ext_modules=[
        CppExtension(
            name='fused_graph_ops',
            sources=['fused_graph_ops.cpp'],
            extra_compile_args=['/O2', '/openmp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)