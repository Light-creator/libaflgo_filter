#include <AFLGoLinker/IntermediateInjectionFixup.hpp>
#include <Analysis/TargetDetection.hpp>

#include <llvm/IR/PassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ADT/DenseMap.h>

#include <llvm/IR/Dominators.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>

#include <llvm/Support/FileSystem.h>
#include <system_error>

using namespace llvm;

PreservedAnalyses
AFLGoIntermediateInjectionFixupPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto &Context = M.getContext();
  auto TraceIntermediate = M.getOrInsertFunction(
      "__aflgo_trace_intermediate", Type::getVoidTy(Context),
      Type::getInt32Ty(Context));
  uint32_t intermediateCounter = 0;
  DenseMap<BasicBlock *, uint32_t> node_ids;

  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  std::error_code EC;
  raw_fd_ostream outFile("intermediate_nodes.csv", EC, sys::fs::OF_Text);
  if (EC) {
    report_fatal_error(Twine("Failed to open intermediate_nodes.csv: ") + EC.message());
  }
  outFile << "target_id,node_id,depth,function,block\n";

  for (auto &F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);

    auto &Targets = FAM.getResult<AFLGoTargetDetectionAnalysis>(F);
    for (auto &TargetPair : Targets.BBs) {
      uint32_t depth = 0;
      auto *CB = TargetPair.second;
      auto target_id = cast<ConstantInt>(CB->getArgOperand(0))->getZExtValue();

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
        if (BB != &F.getEntryBlock()) {
          auto Found = node_ids.find(BB);
          if (Found == node_ids.end()) {
            auto InsertPt = BB->getFirstInsertionPt();
            if (InsertPt != BB->end()) {
              if (intermediateCounter >= 256) {
                report_fatal_error("Intermediate node map capacity (256) exceeded");
              }
              uint32_t ID = intermediateCounter++;
              IRBuilder<> Builder(&*InsertPt);
              Builder.CreateCall(TraceIntermediate, {Builder.getInt32(ID)});
              node_ids[BB] = ID;
              Found = node_ids.find(BB);
              errs() << "Intermediate ID: " << ID
                     << " | function: " << F.getName() << " | block: ";
              BB->printAsOperand(errs(), false);
              errs() << '\n';
            }
          }
          if (Found != node_ids.end()) {
            outFile << target_id << ',' << Found->second << ',' << depth << ',';
            // Quote textual CSV fields, doubling embedded quotation marks.
            auto WriteField = [&](StringRef Text) {
              outFile << '"';
              for (char Ch : Text) {
                if (Ch == '"') outFile << '"';
                outFile << Ch;
              }
              outFile << '"';
            };
            WriteField(F.getName());
            outFile << ',';
            std::string BlockName;
            raw_string_ostream BlockStream(BlockName);
            BB->printAsOperand(BlockStream, false);
            WriteField(BlockStream.str());
            outFile << '\n';
          }
        }
        ++depth;
        nodePtr = nodePtr->getIDom();
      }

    }
  }
  
  outFile.close();
  if (outFile.has_error()) {
    outFile.clear_error();
    report_fatal_error("Failed to write intermediate_nodes.csv");
  }
  return PreservedAnalyses::none();
}
