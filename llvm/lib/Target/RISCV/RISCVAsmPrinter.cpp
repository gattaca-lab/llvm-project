//===-- RISCVAsmPrinter.cpp - RISCV LLVM assembly writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the RISCV assembly language.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "MCTargetDesc/RISCVInstPrinter.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "RISCVTargetMachine.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

STATISTIC(RISCVNumInstrsCompressed,
          "Number of RISC-V Compressed instructions emitted");

namespace {
class RISCVAsmPrinter : public AsmPrinter {
public:
  explicit RISCVAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "RISCV Assembly Printer"; }

  void EmitInstruction(const MachineInstr *MI) override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  void EmitToStreamer(MCStreamer &S, const MCInst &Inst);
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);
  
  typedef std::tuple<unsigned, bool, uint32_t> HwasanMemaccessTuple;
  std::map<HwasanMemaccessTuple, MCSymbol *> HwasanMemaccessSymbols;
  void LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI);
  void EmitHwasanMemaccessSymbols(Module &M);

  // Wrapper needed for tblgenned pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const {
    return LowerRISCVMachineOperandToMCOperand(MO, MCOp, *this);
  }

private:
  void EmitEndOfAsmFile(Module &M) override;
};
}

#define GEN_COMPRESS_INSTR
#include "RISCVGenCompressInstEmitter.inc"
void RISCVAsmPrinter::EmitToStreamer(MCStreamer &S, const MCInst &Inst) {
  MCInst CInst;
  bool Res = compressInst(CInst, Inst, *TM.getMCSubtargetInfo(),
                          OutStreamer->getContext());
  if (Res)
    ++RISCVNumInstrsCompressed;
  AsmPrinter::EmitToStreamer(*OutStreamer, Res ? CInst : Inst);
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "RISCVGenMCPseudoLowering.inc"

void RISCVAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;

  MCInst TmpInst;
  
  if ((MI->getOpcode() == RISCV::HWASAN_CHECK_MEMACCESS) ||
      (MI->getOpcode() == RISCV::HWASAN_CHECK_MEMACCESS_SHORTGRANULES)) {
    LowerHWASAN_CHECK_MEMACCESS(*MI);
    return;
  }

  LowerRISCVMachineInstrToMCInst(MI, TmpInst, *this);
  EmitToStreamer(*OutStreamer, TmpInst);
}

bool RISCVAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                      const char *ExtraCode, raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << RISCVInstPrinter::getRegisterName(RISCV::X0);
        return false;
      }
      break;
    case 'i': // Literal 'i' if operand is not a register.
      if (!MO.isReg())
        OS << 'i';
      return false;
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << RISCVInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  case MachineOperand::MO_BlockAddress: {
    MCSymbol *Sym = GetBlockAddressSymbol(MO.getBlockAddress());
    Sym->print(OS, MAI);
    return false;
  }
  default:
    break;
  }

  return true;
}

bool RISCVAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNo,
                                            const char *ExtraCode,
                                            raw_ostream &OS) {
  if (!ExtraCode) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // For now, we only support register memory operands in registers and
    // assume there is no addend
    if (!MO.isReg())
      return true;

    OS << "0(" << RISCVInstPrinter::getRegisterName(MO.getReg()) << ")";
    return false;
  }

  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVAsmPrinter() {
  RegisterAsmPrinter<RISCVAsmPrinter> X(getTheRISCV32Target());
  RegisterAsmPrinter<RISCVAsmPrinter> Y(getTheRISCV64Target());
}

void RISCVAsmPrinter::LowerHWASAN_CHECK_MEMACCESS(const MachineInstr &MI) {
  Register Reg = MI.getOperand(0).getReg();
  bool IsShort =
      MI.getOpcode() == RISCV::HWASAN_CHECK_MEMACCESS_SHORTGRANULES;
  uint32_t AccessInfo = MI.getOperand(1).getImm();
  MCSymbol *&Sym =
      HwasanMemaccessSymbols[HwasanMemaccessTuple(Reg, IsShort, AccessInfo)];
  if (!Sym) {
    // FIXME: Make this work on non-ELF.
    if (!TM.getTargetTriple().isOSBinFormatELF())
      report_fatal_error("llvm.hwasan.check.memaccess only supported on ELF");

    std::string SymName = "__hwasan_check_x" + utostr(Reg - RISCV::X0) + "_" +
                          utostr(AccessInfo);
    if (IsShort)
      SymName += "_short";
    Sym = OutContext.getOrCreateSymbol(SymName);
  }
  auto Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, OutContext);
  auto Expr = RISCVMCExpr::create(Res, RISCVMCExpr::VK_RISCV_CALL, OutContext);

  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(RISCV::PseudoCALL)
                     .addExpr(Expr));
}

void RISCVAsmPrinter::EmitEndOfAsmFile(Module &M) {
  EmitHwasanMemaccessSymbols(M);
}

void RISCVAsmPrinter::EmitHwasanMemaccessSymbols(Module &M) {
  if (HwasanMemaccessSymbols.empty())
    return;
    

  const Triple &TT = TM.getTargetTriple();
  assert(TT.isOSBinFormatELF());
  std::unique_ptr<MCSubtargetInfo> STI(
      TM.getTarget().createMCSubtargetInfo(TT.str(), "", ""));

  MCSymbol *HwasanTagMismatchV1Sym =
      OutContext.getOrCreateSymbol("__hwasan_tag_mismatch");
  MCSymbol *HwasanTagMismatchV2Sym =
      OutContext.getOrCreateSymbol("__hwasan_tag_mismatch_v2");

  const MCSymbolRefExpr *HwasanTagMismatchV1Ref =
      MCSymbolRefExpr::create(HwasanTagMismatchV1Sym, OutContext);
  const MCSymbolRefExpr *HwasanTagMismatchV2Ref =
      MCSymbolRefExpr::create(HwasanTagMismatchV2Sym, OutContext);

  for (auto &P : HwasanMemaccessSymbols) {
    unsigned Reg = std::get<0>(P.first);
    bool IsShort = std::get<1>(P.first);
    uint32_t AccessInfo = std::get<2>(P.first);
    const MCSymbolRefExpr *HwasanTagMismatchRef =
        IsShort ? HwasanTagMismatchV2Ref : HwasanTagMismatchV1Ref;
    MCSymbol *Sym = P.second;

    OutStreamer->SwitchSection(OutContext.getELFSection(
        ".text.hot", ELF::SHT_PROGBITS,
        ELF::SHF_EXECINSTR | ELF::SHF_ALLOC | ELF::SHF_GROUP, 0,
        Sym->getName()));

    OutStreamer->EmitSymbolAttribute(Sym, MCSA_ELF_TypeFunction);
    OutStreamer->EmitSymbolAttribute(Sym, MCSA_Weak);
    OutStreamer->EmitSymbolAttribute(Sym, MCSA_Hidden);
    OutStreamer->EmitLabel(Sym);

    /* Extract shadow offset from ptr */
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SLLI)
                                     .addReg(RISCV::X6)
                                     .addReg(Reg)
                                     .addImm(8),
                                 *STI);
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SRLI)
                                     .addReg(RISCV::X6)
                                     .addReg(RISCV::X6)
                                     .addImm(12),
                                 *STI);
    /* load shadow tag in X6, X5 contains shadow base */
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::ADD)
                                     .addReg(RISCV::X6)
                                     .addReg(RISCV::X5)
                                     .addReg(RISCV::X6),
                                 *STI);
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::LBU)
                                     .addReg(RISCV::X6)
                                     .addReg(RISCV::X6)
                                     .addImm(0),
                                 *STI);
    /* Extract tag from X5 and compare it with loaded tag from shadow */
    OutStreamer->EmitInstruction(
        MCInstBuilder(RISCV::SRLI)
            .addReg(RISCV::X7)
            .addReg(Reg)
            .addImm(56),
        *STI);
    MCSymbol *HandleMismatchOrPartialSym = OutContext.createTempSymbol();
    OutStreamer->EmitInstruction(
        MCInstBuilder(RISCV::BNE)
            .addReg(RISCV::X7)
            .addReg(RISCV::X6)
            .addExpr(MCSymbolRefExpr::create(HandleMismatchOrPartialSym,
                                             OutContext)),
        *STI);
    MCSymbol *ReturnSym = OutContext.createTempSymbol();
    OutStreamer->EmitLabel(ReturnSym);
    OutStreamer->EmitInstruction(
        MCInstBuilder(RISCV::JALR)
                .addReg(RISCV::X0)
                .addReg(RISCV::X1)
                .addImm(0),
                *STI);
    OutStreamer->EmitLabel(HandleMismatchOrPartialSym);

    if (IsShort) {
      OutStreamer->EmitInstruction(MCInstBuilder(RISCV::ADDI)
                                       .addReg(RISCV::X28)
                                       .addReg(RISCV::X0)
                                       .addImm(16),
                                   *STI);
      MCSymbol *HandleMismatchSym = OutContext.createTempSymbol();
      OutStreamer->EmitInstruction(
          MCInstBuilder(RISCV::BGEU)
              .addReg(RISCV::X6)
              .addReg(RISCV::X28)
              .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
          *STI);

      OutStreamer->EmitInstruction(
          MCInstBuilder(RISCV::ANDI)
              .addReg(RISCV::X28)
              .addReg(Reg)
              .addImm(0xF),
          *STI);
      unsigned Size = 1 << (AccessInfo & 0xf);
      if (Size != 1)
        OutStreamer->EmitInstruction(MCInstBuilder(RISCV::ADDI)
                                         .addReg(RISCV::X28)
                                         .addReg(RISCV::X28)
                                         .addImm(Size - 1),
                                     *STI);
      OutStreamer->EmitInstruction(
          MCInstBuilder(RISCV::BGE)
              .addReg(RISCV::X28)
              .addReg(RISCV::X6)
              .addExpr(MCSymbolRefExpr::create(HandleMismatchSym, OutContext)),
          *STI);

      OutStreamer->EmitInstruction(
          MCInstBuilder(RISCV::ORI)
              .addReg(RISCV::X6)
              .addReg(Reg)
              .addImm(0xF),
          *STI);
      OutStreamer->EmitInstruction(MCInstBuilder(RISCV::LBU)
                                       .addReg(RISCV::X6)
                                       .addReg(RISCV::X6)
                                       .addImm(0),
                                   *STI);
      OutStreamer->EmitInstruction(
          MCInstBuilder(RISCV::BEQ)
              .addReg(RISCV::X6)
              .addReg(RISCV::X7)
              .addExpr(MCSymbolRefExpr::create(ReturnSym, OutContext)),
          *STI);

      OutStreamer->EmitLabel(HandleMismatchSym);
    }

    // +---------------------------------+
    // | Return address (x30) for caller |
    // | of __hwasan_check_*.            |
    // +---------------------------------+
    // | Frame address (x29) for caller  |
    // | of __hwasan_check_*             |
    // +---------------------------------+ <-- [SP + 232]
    // |              ...                |
    // |                                 |
    // | Stack frame space for x2 - x28. |
    // |                                 |
    // |              ...                |
    // +---------------------------------+ <-- [SP + 16]
    // |                                 |
    // | Saved x1, as __hwasan_check_*   |
    // | clobbers it.                    |
    // +---------------------------------+
    // | Saved x0, likewise above.       |
    // +---------------------------------+ <-- [x30 / SP]
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::ADDI)
                                     .addReg(RISCV::X2)
                                     .addReg(RISCV::X2)
                                     .addImm(-256 - 8 * 2),
                                 *STI);

    // x10 - arg #0 (a0)
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SD)
                                     .addReg(RISCV::X10)
                                     .addReg(RISCV::X2)
                                     .addImm(8 * 10),
                                 *STI);
    // x11 - arg #1 (a1)
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SD)
                                     .addReg(RISCV::X11)
                                     .addReg(RISCV::X2)
                                     .addImm(8 * 11),
                                 *STI);

    // x8 - fp
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SD)
                                     .addReg(RISCV::X8)
                                     .addReg(RISCV::X2)
                                     .addImm(32 * 8),
                                 *STI);
    // x1 - ra (in our case it is corrupted most likely)
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::SD)
                                     .addReg(RISCV::X1)
                                     .addReg(RISCV::X2)
                                     .addImm(33 * 8),
                                 *STI);

    if (Reg != RISCV::X10)
      OutStreamer->EmitInstruction(MCInstBuilder(RISCV::OR)
                                       .addReg(RISCV::X10)
                                       .addReg(RISCV::X0)
                                       .addReg(Reg),
                                   *STI);
    OutStreamer->EmitInstruction(MCInstBuilder(RISCV::ADDI)
                                     .addReg(RISCV::X11)
                                     .addReg(RISCV::X0)
                                     .addImm(AccessInfo),
                                 *STI);

    // Intentionally load the GOT entry and branch to it, rather than possibly
    // late binding the function, which may clobber the registers before we have
    // a chance to save them.


    RISCVMCExpr::VariantKind VKHi;
    unsigned SecondOpcode;
    // FIXME: Should check .option (no)pic when implemented
    if (OutContext.getObjectFileInfo()->isPositionIndependent()) {
        SecondOpcode = RISCV::LD;
        VKHi = RISCVMCExpr::VK_RISCV_GOT_HI;
    } else {
        SecondOpcode = RISCV::ADDI;
        VKHi = RISCVMCExpr::VK_RISCV_PCREL_HI;
    }
    auto ExprHi = RISCVMCExpr::create(HwasanTagMismatchRef, VKHi, OutContext);

    MCSymbol *TmpLabel = OutContext.createTempSymbol("pcrel_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
    OutStreamer->EmitLabel(TmpLabel);
    const MCExpr *ExprLo = RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, OutContext),
                                               RISCVMCExpr::VK_RISCV_PCREL_LO, OutContext);

    OutStreamer->EmitInstruction(
        MCInstBuilder(RISCV::AUIPC)
            .addReg(RISCV::X6)
            .addExpr(ExprHi),
        *STI);
    OutStreamer->EmitInstruction(
        MCInstBuilder(SecondOpcode)
            .addReg(RISCV::X6)
            .addReg(RISCV::X6)
            .addExpr(ExprLo),
        *STI);

    OutStreamer->EmitInstruction(
        MCInstBuilder(RISCV::JALR).addReg(RISCV::X0).addReg(RISCV::X6).addImm(0), *STI);
  }
}
