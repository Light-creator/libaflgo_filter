#include <AFLGoLinker/IntermediateInjectionFixup.hpp>
#include <Analysis/TargetDetection.hpp>

#include <llvm/IR/PassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ADT/SmallPtrSet.h>

#include <llvm/IR/Dominators.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>

using namespace llvm;

PreservedAnalyses
AFLGoIntermediateInjectionFixupPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &Context = M.getContext();
  auto TraceIntermediate = M.getOrInsertFunction(
      "__aflgo_trace_intermediate", Type::getVoidTy(Context),
      Type::getInt32Ty(Context));
  uint32_t intermediateCounter = 0;
  auto dominators = SmallPtrSet<BasicBlock*, 32>();

  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (auto &F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);

    auto &Targets = FAM.getResult<AFLGoTargetDetectionAnalysis>(F);
    for (auto &TargetPair : Targets.BBs) {

      auto* targetBlock = TargetPair.first;
      errs() << "Found target in function: " << F.getName() << " | block: ";
      targetBlock->printAsOperand(errs(), false);
      errs() << '\n';

      auto* node = DT.getNode(targetBlock);
      if(node == nullptr) {
        continue;
      }

      auto* nodePtr = node->getIDom();
      while(nodePtr != nullptr) {
        auto* BB = nodePtr->getBlock();
        if (BB != &F.getEntryBlock() && !dominators.contains(BB)) {
          auto InsertPt = BB->getFirstInsertionPt();
          if (InsertPt != BB->end()) {
            if (intermediateCounter >= 256) {
              report_fatal_error("Intermediate node map capacity (256) exceeded");
            }

            IRBuilder<> Builder(&*InsertPt);
            Builder.CreateCall(TraceIntermediate,
                               {Builder.getInt32(intermediateCounter)});
            dominators.insert(BB);

            errs() << "Intermediate ID: " << intermediateCounter
                   << " | function: " << F.getName() << " | block: ";
            BB->printAsOperand(errs(), false);
            errs() << '\n';
            ++intermediateCounter;
          }
        }
        nodePtr = nodePtr->getIDom();
      }

    }
  }

  return PreservedAnalyses::none();
}
