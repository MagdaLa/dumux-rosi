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
 * \brief Volume averaged quantities required by the RichardsTwoC model.
 */
#ifndef DUMUX_RICHARDS_2C_VOLUME_VARIABLES_HH
#define DUMUX_RICHARDS_2C_VOLUME_VARIABLES_HH

#include "richards2cproperties.hh"

#include <dumux/implicit/volumevariables.hh>
#include <dumux/material/fluidstates/compositional.hh>

namespace Dumux
{

/*!
 * \ingroup RichardsTwoCModel
 * \ingroup ImplicitVolumeVariables
 * \brief Volume averaged quantities required by the RichardsTwoC model.
 *
 * This contains the quantities which are are constant within a finite
 * volume in the RichardsTwoC model
 */
template <class TypeTag>
class RichardsTwoCVolumeVariables : public ImplicitVolumeVariables<TypeTag>
{
    typedef ImplicitVolumeVariables<TypeTag> ParentType;

    typedef typename GET_PROP_TYPE(TypeTag, VolumeVariables) Implementation;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, Problem) Problem;
    typedef typename GET_PROP_TYPE(TypeTag, FluidSystem) FluidSystem;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLaw) MaterialLaw;
    typedef typename GET_PROP_TYPE(TypeTag, MaterialLawParams) MaterialLawParams;
    typedef typename GET_PROP_TYPE(TypeTag, FVElementGeometry) FVElementGeometry;
    typedef typename GET_PROP_TYPE(TypeTag, PrimaryVariables) PrimaryVariables;

    static const bool usePH = GET_PROP_VALUE(TypeTag, UsePH);
    static const bool useMoles = GET_PROP_VALUE(TypeTag, UseMoles);

    typedef typename GET_PROP_TYPE(TypeTag, Indices) Indices;

    //indices of primary variables
    enum{hIdx = Indices::hIdx,
         pwIdx = Indices::pwIdx,
         massOrMoleFracIdx = Indices::massOrMoleFracIdx
    };

    enum{
         wPhaseIdx = Indices::wPhaseIdx,
         nPhaseIdx = Indices::nPhaseIdx,
         phaseCompIdx = Indices::phaseCompIdx,
         transportCompIdx = Indices::transportCompIdx
    };

    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;
    enum { dim = GridView::dimension };
    enum { dimWorld = GridView::dimensionworld };


    typedef typename GridView::template Codim<0>::Entity Element;
    typedef Dune::FieldVector<Scalar,dim> DimVector;
    typedef Dune::FieldVector<Scalar,dimWorld> GlobalPosition;

public:
    //! The type returned by the fluidState() method
    typedef Dumux::CompositionalFluidState<Scalar, FluidSystem> FluidState;

    /*!
     * \copydoc ImplicitVolumeVariables::update
     */
    void update(const PrimaryVariables &priVars,
                const Problem &problem,
                const Element &element,
                const FVElementGeometry &fvGeometry,
                const int scvIdx,
                const bool isOldSol)
    {
        assert(!FluidSystem::isLiquid(nPhaseIdx));

        ParentType::update(priVars,
                           problem,
                           element,
                           fvGeometry,
                           scvIdx,
                           isOldSol);

        completeFluidState(priVars, problem, element, fvGeometry, scvIdx, fluidState_);

        //////////
        // specify the other parameters
        //////////
        const MaterialLawParams &matParams =
            problem.spatialParams().materialLawParams(element, fvGeometry, scvIdx);
        relativePermeabilityWetting_ =
            MaterialLaw::krw(matParams,
                             fluidState_.saturation(wPhaseIdx));

        porosity_ = problem.spatialParams().porosity(element, fvGeometry, scvIdx);

        dispersivity_ = problem.spatialParams().dispersivity(element, fvGeometry, scvIdx);

        // Second instance of a parameter cache.
        // Could be avoided if diffusion coefficients also
        // became part of the fluid state.
        typename FluidSystem::ParameterCache paramCache;
        paramCache.updatePhase(fluidState_, wPhaseIdx);

        diffCoeff_ = FluidSystem::binaryDiffusionCoefficient(fluidState_,
                                                             paramCache,
                                                             wPhaseIdx,
                                                             phaseCompIdx,
                                                             transportCompIdx);

        Valgrind::CheckDefined(porosity_);
        Valgrind::CheckDefined(dispersivity_);
        Valgrind::CheckDefined(diffCoeff_);

        // energy related quantities not contained in the fluid state
        asImp_().updateEnergy_(priVars, problem, element, fvGeometry, scvIdx, isOldSol);
    }

    /*!
     * \copydoc ImplicitModel::completeFluidState
     */
    static void completeFluidState(const PrimaryVariables& priVars,
                                   const Problem& problem,
                                   const Element& element,
                                   const FVElementGeometry& fvGeometry,
                                   const int scvIdx,
                                   FluidState& fluidState)
    {
        // temperature
        Scalar t = Implementation::temperature_(priVars, problem, element,
                                                fvGeometry, scvIdx);
        fluidState.setTemperature(t);

        const MaterialLawParams &matParams =
                problem.spatialParams().materialLawParams(element, fvGeometry, scvIdx);

        // pressure head formulation
        if (usePH){
            Scalar pw = (0.01*priVars[hIdx])*1000.0 * 9.81 ;
            fluidState.setPressure(wPhaseIdx, pw);
            fluidState.setPressure(nPhaseIdx, 0.0);

            // saturations
            Scalar sw = MaterialLaw::sw(matParams,-fluidState.pressure(wPhaseIdx));
            fluidState.setSaturation(wPhaseIdx, sw);
            fluidState.setSaturation(nPhaseIdx, 1 - sw);
        }
        else{ //pressure formulation

            Scalar pnRef = problem.referencePressure(element, fvGeometry, scvIdx);

            Scalar minPc = MaterialLaw::pc(matParams, 1.0);
            fluidState.setPressure(wPhaseIdx, priVars[pwIdx]);
            fluidState.setPressure(nPhaseIdx, std::max(pnRef, priVars[pwIdx] + minPc));

            // saturations
            Scalar sw = MaterialLaw::sw(matParams, fluidState.pressure(nPhaseIdx) - fluidState.pressure(wPhaseIdx));
            fluidState.setSaturation(wPhaseIdx, sw);
            fluidState.setSaturation(nPhaseIdx, 1 - sw);
        }
        // density and viscosity
        typename FluidSystem::ParameterCache paramCache;
        paramCache.updateAll(fluidState);
        fluidState.setDensity(wPhaseIdx, FluidSystem::density(fluidState, paramCache, wPhaseIdx));
        fluidState.setDensity(nPhaseIdx, 1e-10);

        fluidState.setViscosity(wPhaseIdx, FluidSystem::viscosity(fluidState, paramCache, wPhaseIdx));
        fluidState.setViscosity(nPhaseIdx, 1e-10);

        // compute and set the enthalpy
        fluidState.setEnthalpy(wPhaseIdx, Implementation::enthalpy_(fluidState, paramCache, wPhaseIdx));
        fluidState.setEnthalpy(nPhaseIdx, Implementation::enthalpy_(fluidState, paramCache, nPhaseIdx));

        Scalar x1 = priVars[massOrMoleFracIdx]; //mole or mass fraction of component 1
        if(!useMoles) //mass-fraction formulation
        {
            // convert mass to mole fractions
            Scalar M0 = FluidSystem::molarMass(phaseCompIdx);
            Scalar M1 = FluidSystem::molarMass(transportCompIdx);
            //meanMolarMass if x1_ is a massfraction
            Scalar meanMolarMass = M0*M1/(M1 + x1*(M0 - M1));

            x1 *= meanMolarMass/M1;
        }
        fluidState.setMoleFraction(wPhaseIdx, phaseCompIdx, 1 - x1);
        fluidState.setMoleFraction(wPhaseIdx, transportCompIdx, x1);

    }

    /*!
     * \brief Return the fluid configuration at the given primary
     *        variables
     */
    const FluidState &fluidState() const
    { return fluidState_; }

    /*!
     * \brief Return molar density \f$\mathrm{[mol/m^3]}\f$ the of the fluid phase.
     */
    Scalar molarDensity() const
    { return fluidState_.molarDensity(wPhaseIdx);}

    /*!
     * \brief Return mole fraction \f$\mathrm{[mol/mol]}\f$ of a component in the phase.
     *
     * \param compIdx The index of the component
     */
    Scalar moleFraction(int compIdx) const
    { return fluidState_.moleFraction(wPhaseIdx, (compIdx==0)?phaseCompIdx:transportCompIdx); }

    /*!
     * \brief Return mass fraction \f$\mathrm{[kg/kg]}\f$ of a component in the phase.
     *
     * \param compIdx The index of the component
     */
    Scalar massFraction(int compIdx) const
    { return fluidState_.massFraction(wPhaseIdx, (compIdx==0)?phaseCompIdx:transportCompIdx); }

    /*!
     * \brief Return concentration \f$\mathrm{[mol/m^3]}\f$  of a component in the phase.
     *
     * \param compIdx The index of the component
     */
    Scalar molarity(int compIdx) const
    { return fluidState_.molarity(wPhaseIdx, (compIdx==0)?phaseCompIdx:transportCompIdx); }

    /*!
     * \brief Return the binary diffusion coefficient \f$\mathrm{[m^2/s]}\f$ in the fluid.
     */
    Scalar diffCoeff() const
    { return diffCoeff_; }

    /*!
     * \brief Returns the dispersivity of the fluid's streamlines.
     */
    const GlobalPosition &dispersivity() const
    { return dispersivity_; }

    /*!
     * \brief Returns the average porosity [] within the control volume.
     *
     * The porosity is defined as the ratio of the pore space to the
     * total volume, i.e. \f[ \Phi := \frac{V_{pore}}{V_{pore} + V_{rock}} \f]
     */
    Scalar porosity() const
    { return porosity_; }

    /*!
     * \brief Returns the average absolute saturation [] of a given
     *        fluid phase within the finite volume.
     *
     * The saturation of a fluid phase is defined as the fraction of
     * the pore volume filled by it, i.e.
     * \f[ S_\alpha := \frac{V_\alpha}{V_{pore}} = \phi \frac{V_\alpha}{V} \f]
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar saturation(const int phaseIdx) const
    { return fluidState_.saturation(phaseIdx); }

    /*!
     * \brief Returns the average mass density \f$\mathrm{[kg/m^3]}\f$ of a given
     *        fluid phase within the control volume.
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar density(const int phaseIdx) const
    { return fluidState_.density(phaseIdx); }

    /*!
     * \brief Returns the effective pressure \f$\mathrm{[Pa]}\f$ of a given phase within
     *        the control volume.
     *
     * For the non-wetting phase (i.e. the gas phase), we assume
     * infinite mobility, which implies that the non-wetting phase
     * pressure is equal to the finite volume's reference pressure
     * defined by the problem.
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar pressure(const int phaseIdx) const
    { return fluidState_.pressure(phaseIdx); }

    /*!
     * \brief Returns average temperature \f$\mathrm{[K]}\f$ inside the control volume.
     *
     * Note that we assume thermodynamic equilibrium, i.e. the
     * temperature of the rock matrix and of all fluid phases are
     * identical.
     */
    Scalar temperature() const
    { return fluidState_.temperature(); }

    /*!
     * \brief Returns the effective mobility \f$\mathrm{[1/(Pa*s)]}\f$ of a given phase within
     *        the control volume.
     *
     * The mobility of a fluid phase is defined as the relative
     * permeability of the phase (given by the chosen material law)
     * divided by the dynamic viscosity of the fluid, i.e.
     * \f[ \lambda_\alpha := \frac{k_{r\alpha}}{\mu_\alpha} \f]
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar mobility(const int phaseIdx) const
    { return relativePermeability(phaseIdx)/fluidState_.viscosity(phaseIdx); }

    /*!
     * \brief Return the dynamic viscosity \f$\mathrm{[Pa*s]}\f$ of a given phase
     *        within the control volume.
     */
    Scalar viscosity(const int phaseIdx) const
    { return fluidState_.viscosity(phaseIdx); }

    /*!
     * \brief Returns relative permeability [-] of a given phase within
     *        the control volume.
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar relativePermeability(const int phaseIdx) const
    {
        if (phaseIdx == wPhaseIdx)
            return relativePermeabilityWetting_;
        return 1;
    }

    /*!
     * \brief Returns the effective capillary pressure \f$\mathrm{[Pa]}\f$ within the
     *        control volume.
     *
     * The capillary pressure is defined as the difference in
     * pressures of the non-wetting and the wetting phase, i.e.
     * \f[ p_c = p_n - p_w \f]
     */
    Scalar capillaryPressure() const
    {
        // pressure head formulation
        if (usePH)
            return -fluidState_.pressure(wPhaseIdx);
        else // pressure  formulation
            return fluidState_.pressure(nPhaseIdx) - fluidState_.pressure(wPhaseIdx);
    }

    /*!
     * \brief Returns the pressureHead \f$\mathrm{[cm]}\f$ of a given phase within
     *        the control volume.
     *
     * For the non-wetting phase (i.e. the gas phase), we assume
     * infinite mobility, which implies that the non-wetting phase
     * pressure is equal to the finite volume's reference pressure
     * defined by the problem.
     *
     * \param phaseIdx The index of the fluid phase
     */
    Scalar pressureHead(const int phaseIdx) const
    {
        // pressure head formulation
        if (usePH)
            return (100.) *(fluidState_.pressure(phaseIdx))/ fluidState_.density(phaseIdx)/ 9.81;
        else // pressure  formulation
            return (100.) *(fluidState_.pressure(phaseIdx) -1e5)/ fluidState_.density(phaseIdx)/ 9.81;
    }

    /*!
     * \brief Returns the water content
     *        fluid phase within the finite volume.
     *
     * The water content is defined as the fraction of
     * the saturation devided by the porosity

     * \param phaseIdx The index of the fluid phase
     */
    Scalar waterContent (const int phaseIdx) const
    { return fluidState_.saturation(phaseIdx)* porosity_; }

protected:
    static Scalar temperature_(const PrimaryVariables &primaryVariables,
                            const Problem& problem,
                            const Element &element,
                            const FVElementGeometry &fvGeometry,
                            const int scvIdx)
    {
        return problem.temperatureAtPos(fvGeometry.subContVol[scvIdx].global);
    }

    template<class ParameterCache>
    static Scalar enthalpy_(const FluidState& fluidState,
                            const ParameterCache& paramCache,
                            int phaseIdx)
    {
        return 0;
    }

    /*!
     * \brief Called by update() to compute the energy related quantities
     */
    void updateEnergy_(const PrimaryVariables &priVars,
                       const Problem &problem,
                       const Element &element,
                       const FVElementGeometry &fvGeometry,
                       const int scvIdx,
                       const bool isOldSol)
    { }

    FluidState fluidState_;
    Scalar relativePermeabilityWetting_;
    Scalar porosity_;
    GlobalPosition dispersivity_;
    Scalar diffCoeff_;

private:
    Implementation &asImp_()
    { return *static_cast<Implementation*>(this); }

    const Implementation &asImp_() const
    { return *static_cast<const Implementation*>(this); }
};

}

#endif