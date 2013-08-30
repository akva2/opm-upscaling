//==============================================================================
//!
//! \file elasticity_preconditioners.hpp
//!
//! \date Aug 30 2013
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Preconditioners for elasticity upscaling
//!
//==============================================================================

#include "config.h"

#include "elasticity_preconditioners.hpp"

namespace Opm {
namespace Elasticity {

std::shared_ptr<FastAMG::type> FastAMG::setup(int pre, int post, int target,
                                              int zcells,
                                              std::shared_ptr<Operator>& op,
                                              const Dune::CpGrid& gv,
                                              ASMHandler<Dune::CpGrid>& A,
                                              bool& copy)
{
  AMG1<JACSmoother>::Criterion crit;
  crit.setCoarsenTarget(target);
  crit.setGamma(1);
  crit.setDefaultValuesIsotropic(3, zcells);

  std::cout << "\t collapsing 2x2x" << zcells << " cells per level" << std::endl;
  copy = true;
  return std::shared_ptr<type>(new type(*op, crit));
}

Schwarz::type* Schwarz::setup2(int pre, int post, int target,
                               int zcells, std::shared_ptr<Operator>& op,
                               const Dune::CpGrid& gv,
                               ASMHandler<Dune::CpGrid>& A, bool& copy)
{
  const int cps = 2;
  Schwarz::type::subdomain_vector rows;
  int nel1 = gv.logicalCartesianSize()[0];
  int nel2 = gv.logicalCartesianSize()[1];
  int nel3 = gv.logicalCartesianSize()[2];
  rows.resize(nel1/cps*nel2/cps);

  // invert compressed cell array
  std::vector<int> globalActive(nel1*nel2*nel3, -1);
  for (size_t i=0;i<gv.globalCell().size();++i)
    globalActive[gv.globalCell()[i]] = i;

  auto set = gv.leafView().indexSet();
  for (int i=0;i<nel2;++i) {
    for (int j=0;j<nel1;++j) {
      for (int k=0;k<nel3;++k) {
        if (globalActive[k*nel1*nel2+i*nel1+j] > -1) {
          auto it = gv.leafView().begin<0>();
          for (int l=0;l<globalActive[k*nel1*nel2+i*nel1+j];++l)
            ++it;
          // loop over nodes
          for (int n=0;n<8;++n) {
            int idx = set.subIndex(*it, n, 3);
            for (int m=0;m<3;++m) {
              const MPC* mpc = A.getMPC(idx, m);
              if (mpc) {
                for (size_t q=0;q<mpc->getNoMaster();++q) {
                  int idx2 = A.getEquationForDof(mpc->getMaster(q).node, m);
                  if (idx2 > -1)
                    rows[(i/cps)*(nel1/cps)+j/cps].insert(idx2);
                }
              } else {
                if (A.getEquationForDof(idx, m) > -1)
                  rows[i/cps*nel1/cps+j/cps].insert(A.getEquationForDof(idx, m));
              }
            }
          }
        }
      }
    }
  }

  copy = false;
  return new type(op->getmat(), rows, 1.0, false);
}

}
}
