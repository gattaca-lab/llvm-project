//===-- RISCV.td - Describe the RISCV Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// RISC-V subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;

def FeatureStdExtM
    : SubtargetFeature<"m", "HasStdExtM", "true",
                       "'M' (Integer Multiplication and Division)">;
def HasStdExtM : Predicate<"Subtarget->hasStdExtM()">,
                           AssemblerPredicate<"FeatureStdExtM",
                           "'M' (Integer Multiplication and Division)">;

def FeatureStdExtA
    : SubtargetFeature<"a", "HasStdExtA", "true",
                       "'A' (Atomic Instructions)">;
def HasStdExtA : Predicate<"Subtarget->hasStdExtA()">,
                           AssemblerPredicate<"FeatureStdExtA",
                           "'A' (Atomic Instructions)">;

def FeatureStdExtF
    : SubtargetFeature<"f", "HasStdExtF", "true",
                       "'F' (Single-Precision Floating-Point)">;
def HasStdExtF : Predicate<"Subtarget->hasStdExtF()">,
                           AssemblerPredicate<"FeatureStdExtF",
                           "'F' (Single-Precision Floating-Point)">;

def FeatureStdExtD
    : SubtargetFeature<"d", "HasStdExtD", "true",
                       "'D' (Double-Precision Floating-Point)",
                       [FeatureStdExtF]>;
def HasStdExtD : Predicate<"Subtarget->hasStdExtD()">,
                           AssemblerPredicate<"FeatureStdExtD",
                           "'D' (Double-Precision Floating-Point)">;

def FeatureStdExtC
    : SubtargetFeature<"c", "HasStdExtC", "true",
                       "'C' (Compressed Instructions)">;
def HasStdExtC : Predicate<"Subtarget->hasStdExtC()">,
                           AssemblerPredicate<"FeatureStdExtC",
                           "'C' (Compressed Instructions)">;

def FeatureRVCHints
    : SubtargetFeature<"rvc-hints", "EnableRVCHintInstrs", "true",
                       "Enable RVC Hint Instructions.">;
def HasRVCHints : Predicate<"Subtarget->enableRVCHintInstrs()">,
                            AssemblerPredicate<"FeatureRVCHints",
                            "RVC Hint Instructions">;

def Feature64Bit
    : SubtargetFeature<"64bit", "HasRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<"Feature64Bit",
                       "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<"!Feature64Bit",
                       "RV32I Base Instruction Set">;

def RV64           : HwMode<"+64bit">;
def RV32           : HwMode<"-64bit">;

def FeatureRV32E
    : SubtargetFeature<"e", "IsRV32E", "true",
                       "Implements RV32E (provides 16 rather than 32 GPRs)">;
def IsRV32E : Predicate<"Subtarget->isRV32E()">,
                        AssemblerPredicate<"FeatureRV32E">;

def FeatureRelax
    : SubtargetFeature<"relax", "EnableLinkerRelax", "true",
                       "Enable Linker relaxation.">;

foreach i = {1-31} in
    def FeatureReserveX#i :
        SubtargetFeature<"reserve-x"#i, "UserReservedRegister[RISCV::X"#i#"]",
                         "true", "Reserve X"#i>;

//===----------------------------------------------------------------------===//
// Named operands for CSR instructions.
//===----------------------------------------------------------------------===//

include "RISCVSystemOperands.td"

//===----------------------------------------------------------------------===//
// Registers, calling conventions, instruction descriptions.
//===----------------------------------------------------------------------===//

include "RISCVRegisterInfo.td"
include "RISCVCallingConv.td"
include "RISCVInstrInfo.td"
include "RISCVRegisterBanks.td"

//===----------------------------------------------------------------------===//
// RISC-V processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic-rv32", NoSchedModel, [FeatureRVCHints]>;

def : ProcessorModel<"generic-rv64", NoSchedModel, [Feature64Bit,
                     FeatureRVCHints]>;

//===----------------------------------------------------------------------===//
// Define the RISC-V target.
//===----------------------------------------------------------------------===//

def RISCVInstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def RISCVAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
}

def RISCVAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def RISCV : Target {
  let InstructionSet = RISCVInstrInfo;
  let AssemblyParsers = [RISCVAsmParser];
  let AssemblyWriters = [RISCVAsmWriter];
  let AllowRegisterRenaming = 1;
}
