//==============================================================================
//!
//! \file upscale_elasticity.cpp
//!
//! \date Nov 9 2011
//!
//! \author Arne Morten Kvarving / SINTEF
//!
//! \brief Elasticity upscaling on cornerpoint grids
//!
//==============================================================================
#ifdef HAVE_CONFIG_H
# include "config.h"     
#endif

#include <opm/core/utility/have_boost_redef.hpp>

#include <iostream>
#include <unistd.h>
#include <cstring>
#include <dune/common/exceptions.hh> // We use exceptions
#include <opm/core/utility/StopWatch.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/istl/io.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/paamg/amg.hh>

#if HAVE_OPENMP
#include <omp.h>
#endif

typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,3,3> > Matrix;
typedef Dune::BlockVector<Dune::FieldVector<double,3> > Vector;

//! \brief Linear solver
Dune::InverseOperator<Vector, Vector>* solver;

//! \brief The smoother used in the AMG
typedef Dune::SeqSSOR<Matrix, Vector, Vector> Smoother;

//! \brief The coupling metric used in the AMG
typedef Dune::Amg::RowSum CouplingMetric;

//! \brief The coupling criterion used in the AMG
typedef Dune::Amg::SymmetricCriterion<Matrix, CouplingMetric> CritBase;

//! \brief The coarsening criterion used in the AMG
typedef Dune::Amg::CoarsenCriterion<CritBase> Criterion;

//! \brief A linear operator
typedef Dune::MatrixAdapter<Matrix,Vector,Vector> Operator;

//! \brief A preconditioner for an elasticity operator
typedef Dune::Amg::AMG<Operator, Vector, Smoother> ElasticityAMG;

//! \brief Main driver
int main(int argc, char** argv)
{
  try {
    Opm::parameter::ParameterGroup param(argc, argv, false);
    std::string matrix = param.getDefault<std::string>("matrix","A.mm");
    std::string vector = param.getDefault<std::string>("vector","b.mm");
    int coarsen_target = param.getDefault("coarsen_target", 1000);
    double relax_factor = param.getDefault<double>("relax_factor", 1.0);
    double damp_factor = param.getDefault<double>("damp_factor", 1.6);
    double gamma = param.getDefault<double>("gamma", 1.0);
    int presteps = param.getDefault("presteps", 1);
    int poststeps = param.getDefault("poststeps", 1);
    int agdim = param.getDefault("agdim", 2);
    bool aniso = param.getDefault<bool>("anisotropic", false);

    Matrix A;
    Vector b;
    std::ifstream a;
    a.open(matrix.c_str());
    Dune::readMatrixMarket(A, a);

    a.close();
    a.open(vector.c_str());
    Dune::readMatrixMarket(b, a);

    Criterion crit;
    ElasticityAMG::SmootherArgs args;
    args.relaxationFactor = relax_factor;
    crit.setCoarsenTarget(coarsen_target);
    crit.setGamma(gamma);
    crit.setNoPreSmoothSteps(presteps);
    crit.setNoPostSmoothSteps(poststeps);
    crit.setProlongationDampingFactor(damp_factor);
    if (aniso)
      crit.setDefaultValuesAnisotropic(3, agdim);
    else
      crit.setDefaultValuesIsotropic(3, agdim);

    Operator op(A);
    ElasticityAMG upre(op, crit, args);

    Dune::CGSolver<Vector> solver(op, upre, 1e-8, 1000, 2);
    Vector sol;
    sol.resize(b.size());
    Dune::InverseOperatorResult r;
    solver.apply(sol, b, r);

    return 0;
  }
  catch (Dune::Exception &e) {
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
  return 1;
}
