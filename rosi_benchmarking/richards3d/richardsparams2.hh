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
 * \ingroup RichardsTests
 * \brief spatial parameters for the RichardsLensProblem
 */
#ifndef RICHARDS_PARAMS
#define RICHARDS_PARAMS

#include <dumux/material/spatialparams/fv.hh>
#include <dumux/material/fluidmatrixinteractions/2p/regularizedvangenuchten.hh>
#include <dumux/material/fluidmatrixinteractions/2p/efftoabslaw.hh>

#include <dumux/porousmediumflow/richards/model.hh>


namespace Dumux
{
/*!
 * \ingroup RichardsTests
 * \brief spatial parameters for the RichardsLensProblem
 */
// forward declaration
template<class TypeTag>
class RichardsParams;

namespace Properties
{
// The spatial parameters TypeTag
NEW_TYPE_TAG(RichardsParams);

// Set the spatial parameters
SET_TYPE_PROP(RichardsParams, SpatialParams, RichardsParams<TypeTag>);

// Set the material law
SET_PROP(RichardsParams, MaterialLaw)
{
private:
    // define the material law which is parameterized by effective
    // saturations
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);
public:
    // define the material law parameterized by absolute saturations
    using type = EffToAbsLaw<RegularizedVanGenuchten<Scalar>>;
};
}

/*!
 * \ingroup RichardsModel
 * \ingroup ImplicitTestProblems
 * \brief The spatial parameters for the RichardsLensProblem
 */
template<class TypeTag>
class RichardsParams : public FVSpatialParams<TypeTag>
{
    using ParentType = FVSpatialParams<TypeTag>;
    using Problem = typename GET_PROP_TYPE(TypeTag, Problem);
    using GridView = typename GET_PROP_TYPE(TypeTag, GridView);
    using Scalar = typename GET_PROP_TYPE(TypeTag, Scalar);

    enum {
        dim=GridView::dimension,
        dimWorld=GridView::dimensionworld
    };

    using GlobalPosition = Dune::FieldVector<Scalar, dimWorld>;
    using MaterialLaw = typename GET_PROP_TYPE(TypeTag, MaterialLaw);
    using MaterialLawParams = typename MaterialLaw::Params;

public:
    // export permeability type
    using PermeabilityType = Scalar;

    /*!
     * \brief Constructor
     *
     * \param gridView The DUNE GridView representing the spatial
     *                 domain of the problem.
     */
    RichardsParams(const Problem& problem)
        : ParentType(problem)
    {

        lensLowerLeft_ = {1.0, 2.0};
        lensUpperRight_ = {4.1, 3.1};

        // residual saturations
        lensMaterialParams_.setSwr(0.18);
        lensMaterialParams_.setSnr(0.0);
        outerMaterialParams_.setSwr(0.05);
        outerMaterialParams_.setSnr(0.0);

        // parameters for the Van Genuchten law
        // alpha and n
        lensMaterialParams_.setVgAlpha(0.00045);
        lensMaterialParams_.setVgn(7.3);
        outerMaterialParams_.setVgAlpha(0.0037);
        outerMaterialParams_.setVgn(4.7);

        lensK_ = 1e-12;
        outerK_ = 5e-12;
    }

    /*!
     * \brief Returns the intrinsic permeability tensor [m^2] at a given location
     *
     * \param globalPos The global position where we evaluate
     */
    PermeabilityType permeabilityAtPos(const GlobalPosition& globalPos) const
    {
        if (isInLens_(globalPos))
            return lensK_;
        return outerK_;
    }

    /*!
     * \brief Returns the porosity [] at a given location
     *
     * \param globalPos The global position where we evaluate
     */
    Scalar porosityAtPos(const GlobalPosition& globalPos) const
    { return 0.4; }

    /*!
     * \brief Returns the parameters for the material law at a given location
     *
     * This method is not actually required by the Richards model, but provided
     * for the convenience of the RichardsLensProblem
     *
     * \param globalPos A global coordinate vector
     */
    const MaterialLawParams& materialLawParamsAtPos(const GlobalPosition &globalPos) const
    {
        if (isInLens_(globalPos))
            return lensMaterialParams_;
        return outerMaterialParams_;
    }

private:
    bool isInLens_(const GlobalPosition &globalPos) const
    {
        for (int i = 0; i < dimWorld; ++i)
            if (globalPos[i] < lensLowerLeft_[i] - eps_ || globalPos[i] > lensUpperRight_[i] + eps_)
                return false;

        return true;
    }

    static constexpr Scalar eps_ = 1.5e-7;

    GlobalPosition lensLowerLeft_;
    GlobalPosition lensUpperRight_;

    Scalar lensK_;
    Scalar outerK_;
    MaterialLawParams lensMaterialParams_;
    MaterialLawParams outerMaterialParams_;
};

} // end namespace Dumux

#endif

