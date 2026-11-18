#pragma once

#include <llvm/Passes/PassBuilder.h>

namespace llvm {

class AFLGoIntermediateInjectionFixupPass
    : public PassInfoMixin<AFLGoIntermediateInjectionFixupPass> {

public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace llvm
