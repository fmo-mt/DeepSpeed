# Copyright (c) Microsoft Corporation.
# Copyright (c) 2024 Cambricon Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401 # type: ignore
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder


class MUSAOpBuilder(OpBuilder):

    def builder(self):
        from torch_musa.utils.musa_extension import MUSAExtension as ExtensionBuilder

        compile_args = {
            'cxx': self.strip_empty_entries(self.cxx_args()),
            'mcc': self.strip_empty_entries(self.mcc_args()),}

        cpp_ext = ExtensionBuilder(name=self.absolute_name(),
                                   sources=self.strip_empty_entries(self.sources()),
                                   include_dirs=self.strip_empty_entries(self.include_paths()),
                                   libraries=self.strip_empty_entries(self.libraries_args()),
                                   extra_compile_args=compile_args)

        return cpp_ext

    def cxx_args(self):
        return ['-O3', '-Wno-reorder']

    def mcc_args(self):
        return ['-O3',]

    def libraries_args(self):
        return []
