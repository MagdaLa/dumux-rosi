// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*****************************************************************************
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/*!
 * \file
 *
 * \brief test for the static periodic network grid manager
 */
#include <config.h>

#include <ctime>
#include <iostream>
#include <bitset>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dumux/common/properties.hh>
#include <dumux/common/parameters.hh>
#include <dumux/common/dumuxmessage.hh>
#include <dumux/common/defaultusagemessage.hh>

#include <dumux/io/vtkoutputmodule.hh>
#include <dumux/assembly/fvassembler.hh>
#include <dumux/linear/seqsolverbackend.hh>

#include <dumux/discretization/cellcentered/tpfa/properties.hh>
#include <dumux/periodic/tpfa/periodicnetworkgridmanager.hh>
#include <dumux/periodic/tpfa/fvgridgeometry.hh>

#include <dumux/porousmediumflow/1p/model.hh>
#include <dumux/material/components/simpleh2o.hh>
#include <dumux/material/components/constant.hh>
#include <dumux/material/fluidsystems/1pliquid.hh>

#include <dumux/growth/problem.hh>

#include "problem.hh"

// Set properties
namespace Dumux {
namespace Properties {

NEW_TYPE_TAG(RootTypeTag, INHERITS_FROM(CCTpfaModel, OneP));

// use simple h2o
SET_PROP(RootTypeTag, FluidSystem)
{
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
    using type = FluidSystems::OnePLiquid<Scalar, Components::SimpleH2O<Scalar>>;
};

// we use foamgrid as grid manager
SET_TYPE_PROP(RootTypeTag, Grid, Dune::FoamGrid<1, 3>);

// set the periodic grid geometry to enable periodic faces via extra connectivities
SET_PROP(RootTypeTag, FVGridGeometry)
{
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using type = PeriodicCCTpfaFVGridGeometry<GridView, /*enableCache=*/true>;
};

// Set the problem property
SET_TYPE_PROP(RootTypeTag, Problem, GrowthModule::GrowthProblemAdapter<RootProblem<TypeTag>>);

// Set the spatial parameters
SET_TYPE_PROP(RootTypeTag, SpatialParams, RootSpatialParams<typename GET_PROP_TYPE(TypeTag, FVGridGeometry),
                                                            typename GET_PROP_TYPE(TypeTag, Scalar)>);

} // end namespace Properties
} // end namespace Dumux

int main(int argc, char** argv) try
{
    using namespace Dumux;

    using TypeTag = TTAG(RootTypeTag);

    // initialize MPI, finalize is done automatically on exit
    const auto& mpiHelper = Dune::MPIHelper::instance(argc, argv);

    // print dumux start message
    if (mpiHelper.rank() == 0)
        DumuxMessage::print(/*firstCall=*/true);

    ////////////////////////////////////////////////////////////
    // parse the command line arguments and input file
    ////////////////////////////////////////////////////////////

    // parse command line arguments
    Parameters::init(argc, argv);

    //////////////////////////////////////////////////////////////////////
    // try to create a grid (from the given grid file or the input file)
    /////////////////////////////////////////////////////////////////////

    const auto periodic = getParam<std::bitset<3>>("Grid.Periodic");
    const auto lowerLeft = getParam<Dune::FieldVector<double, 3>>("Grid.PeriodicBoxLowerLeft");
    const auto upperRight = getParam<Dune::FieldVector<double, 3>>("Grid.PeriodicBoxUpperRight");
    PeriodicNetworkGridManager<3> gridManager(lowerLeft, upperRight, periodic);
    gridManager.init();
    auto gridData = gridManager.getGridData();

    // we compute on the leaf grid view
    const auto& leafGridView = gridManager.grid().leafGridView();

    // create the finite volume grid geometry
    using FVGridGeometry = typename GET_PROP_TYPE(TypeTag, FVGridGeometry);
    auto fvGridGeometry = std::make_shared<FVGridGeometry>(leafGridView);
    const auto periodicConnectivity = gridData->createPeriodicConnectivity(fvGridGeometry->elementMapper(), fvGridGeometry->vertexMapper());
    fvGridGeometry->setExtraConnectivity(periodicConnectivity);
    fvGridGeometry->update();

    // the problem (boundary conditions) and the spatial params
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using SpatialParams = typename Problem::SpatialParams;
    auto spatialParams = std::make_shared<SpatialParams>(fvGridGeometry, gridData);
    auto problem = std::make_shared<Problem>(fvGridGeometry, spatialParams);

    // the solution vector
    using SolutionVector = typename GET_PROP_TYPE(TypeTag, SolutionVector);
    SolutionVector x(fvGridGeometry->numDofs());

    // the grid variables
    using GridVariables = typename GET_PROP_TYPE(TypeTag, GridVariables);
    auto gridVariables = std::make_shared<GridVariables>(problem, fvGridGeometry);
    gridVariables->init(x);

    // intialize the vtk output module
    VtkOutputModule<GridVariables, SolutionVector> vtkWriter(*gridVariables, x, problem->name());
    using VelocityOutput = typename GET_PROP_TYPE(TypeTag, VelocityOutput);
    vtkWriter.addVelocityOutput(std::make_shared<VelocityOutput>(*gridVariables));
    using VtkOutputFields = typename GET_PROP_TYPE(TypeTag, VtkOutputFields);
    VtkOutputFields::init(vtkWriter); //!< Add model specific output fields
    vtkWriter.addField(problem->spatialParams().getRadii(), "radius"); //! Add radius field
    vtkWriter.write(0.0);

    // make assemble and attach linear system
    using Assembler = FVAssembler<TypeTag, DiffMethod::numeric>;
    auto assembler = std::make_shared<Assembler>(problem, fvGridGeometry, gridVariables);
    using JacobianMatrix = typename GET_PROP_TYPE(TypeTag, JacobianMatrix);
    auto A = std::make_shared<JacobianMatrix>();
    auto r = std::make_shared<SolutionVector>();
    assembler->setLinearSystem(A, r);

    Dune::Timer timer;
    // assemble the local jacobian and the residual
    if (mpiHelper.rank() == 0) std::cout << "Assembling linear system ..." << std::flush;
    Dune::Timer assemblyTimer;
    assembler->assembleJacobianAndResidual(x);
    assemblyTimer.stop();
    if (mpiHelper.rank() == 0) std::cout << " took " << assemblyTimer.elapsed() << " seconds." << std::endl;

    // we solve Ax = -r to save update and copy
    (*r) *= -1.0;

    // solve the linear system
    Dune::Timer solverTimer;
    using LinearSolver = SSORCGBackend;
    auto linearSolver = std::make_shared<LinearSolver>();

    if (mpiHelper.rank() == 0) std::cout << "Solving linear system using " + linearSolver->name() + "..." << std::flush;
    linearSolver->solve(*A, x, *r);
    solverTimer.stop();
    if (mpiHelper.rank() == 0) std::cout << " took " << solverTimer.elapsed() << " seconds." << std::endl;

    // the grid variables need to be up to date for subsequent output
    if (mpiHelper.rank() == 0) std::cout << "Updating variables ..." << std::flush;
    Dune::Timer updateTimer;
    gridVariables->update(x);
    updateTimer.stop();
    if (mpiHelper.rank() == 0) std::cout << " took " << updateTimer.elapsed() << std::endl;

    // output result to vtk
    vtkWriter.write(1.0);

    timer.stop();

    const auto& comm = Dune::MPIHelper::getCollectiveCommunication();
    if (mpiHelper.rank() == 0)
        std::cout << "Simulation took " << timer.elapsed() << " seconds on "
                  << comm.size() << " processes.\n"
                  << "The cumulative CPU time was " << timer.elapsed()*comm.size() << " seconds.\n";

    if (mpiHelper.rank() == 0)
        Parameters::print();

    return 0;

}
catch (const Dumux::ParameterException& e)
{
    std::cerr << std::endl << e << " ---> Abort!" << std::endl;
    return 1;
}
catch (const Dune::DGFException& e)
{
    std::cerr << "DGF exception thrown (" << e <<
                 "). Most likely, the DGF file name is wrong "
                 "or the DGF file is corrupted, "
                 "e.g. missing hash at end of file or wrong number (dimensions) of entries."
                 << " ---> Abort!" << std::endl;
    return 2;
}
catch (const Dune::Exception& e)
{
    std::cerr << "Dune reported error: " << e << " ---> Abort!" << std::endl;
    return 3;
}
catch (const std::exception& e)
{
    std::cerr << "Standard library reported error: " << e.what() << " ---> Abort!" << std::endl;
    return 4;
}
