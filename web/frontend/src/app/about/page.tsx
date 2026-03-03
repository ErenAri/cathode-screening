import Link from "next/link";
import { SocialLinks } from "@/components/SocialLinks";
import API_BASE_URL from "@/config";

export default function About() {
    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="fixed w-full top-4 z-50 flex justify-center pointer-events-none">
                <div className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg rounded-full px-8 py-3 flex items-center gap-12 pointer-events-auto">
                    <Link href="/" className="font-bold text-gray-900 text-lg tracking-tight hover:text-blue-600 hover:scale-125 transition-all duration-200">
                        CathodeScreen
                    </Link>
                    <nav className="flex items-center gap-8 text-sm font-medium text-gray-600">
                        <Link href="/predict" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Predict</Link>
                        <Link href="/database" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
                        <Link href="/database#screening-pack" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Screening</Link>
                        <Link href="/discovery" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Discovery</Link>
                        <Link href="/about" className="text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
                    </nav>
                    <div className="pl-8 border-l border-gray-200">
                        <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
                            API <span className="text-[10px]">↗</span>
                        </a>
                    </div>
                </div>
            </header>

            {/* Hero */}
            <section className="bg-gradient-to-b from-slate-50 to-white pt-32 pb-16">
                <div className="max-w-4xl mx-auto px-6 text-center">
                    <h1 className="text-4xl font-bold text-gray-900 mb-4">
                        About CathodeScreen
                    </h1>
                    <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                        A machine learning platform for accelerating thermodynamic stability prediction
                        of lithium-ion battery cathode materials.
                    </p>
                </div>
            </section>

            <main className="max-w-4xl mx-auto px-6 py-12">
                {/* Motivation */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Motivation</h2>
                    <div className="prose prose-gray max-w-none text-gray-600">
                        <p className="mb-4">
                            High-throughput computational screening of cathode materials is bottlenecked by the
                            computational cost of Density Functional Theory (DFT) calculations. A single DFT relaxation
                            can require 10–100 CPU hours, making exhaustive screening of large candidate spaces
                            prohibitively expensive.
                        </p>
                        <p className="mb-4">
                            Furthermore, the majority of randomly sampled candidates are thermodynamically unstable
                            (E<sub>hull</sub> &gt; 50 meV/atom), resulting in wasted computational resources.
                            Machine learning pre-screening offers an efficient filtering mechanism to prioritize
                            candidates before expensive DFT validation.
                        </p>
                    </div>
                </section>

                {/* Approach */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Approach</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p>
                            We employ <strong>MACE-MP-0</strong> (Multi-Atomic Cluster Expansion),
                            a state-of-the-art equivariant foundation model for materials science pretrained on the Materials Project
                            (Batatia et al., 2023). MACE uses higher-order equivariant message passing with multi-body
                            interactions, enabling highly accurate prediction of
                            energy above the convex hull (E<sub>hull</sub>).
                        </p>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 mb-6">
                        <h3 className="font-semibold text-gray-900 mb-4">Model Configuration (v1-Li-Cathode)</h3>
                        <div className="grid md:grid-cols-2 gap-6 text-sm text-gray-700">
                            <div>
                                <h4 className="font-medium text-gray-900 mb-2">Architecture</h4>
                                <ul className="space-y-1">
                                    <li>• MACE-MP-0 &quot;medium&quot; backbone (~3.5M params)</li>
                                    <li>• 5-member deep ensemble (seeds 42–46)</li>
                                    <li>• Frozen backbone, fine-tuned last interaction block</li>
                                    <li>• Custom regression head: quantile outputs (q10, q50, q90) + stability classification (p_stable, p_metastable)</li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-medium text-gray-900 mb-2">Training Details</h4>
                                <ul className="space-y-1">
                                    <li>• Dataset: 17,227 Transition Metal Oxides (TMOs)</li>
                                    <li>• Splitting: SOAP-LOCO (Leave-One-Cluster-Out)</li>
                                    <li>• Target: E<sub>hull</sub> in eV/atom</li>
                                    <li>• Post-hoc symmetric conformal calibration (90% coverage)</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Uncertainty Quantification */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Uncertainty Quantification</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p className="mb-4">
                            We combine three complementary sources of uncertainty following best practices
                            in deep ensemble methodology (Lakshminarayanan et al., 2017):
                        </p>
                    </div>

                    <div className="bg-slate-800 text-slate-100 rounded-lg p-6 font-mono text-sm mb-6">
                        <p className="mb-2 text-slate-400"># Aleatoric (data noise) — per-model quantile regression</p>
                        <p className="mb-4">σ_aleatoric = mean(q90 − q10) across ensemble</p>
                        <p className="mb-2 text-slate-400"># Epistemic (model ignorance) — inter-model disagreement</p>
                        <p className="mb-4">σ_epistemic = std(q50 across 5 members)</p>
                        <p className="mb-2 text-slate-400"># Total uncertainty</p>
                        <p>σ_total = √(σ_aleatoric² + σ_epistemic²)</p>
                    </div>

                    <p className="text-gray-600">
                        Prediction intervals are calibrated via <strong>symmetric conformal prediction</strong> on
                        the validation set, guaranteeing 90% coverage. A conformal delta is added to raw
                        quantile intervals to achieve valid coverage under distribution shift (Vovk et al., 2005).
                    </p>
                </section>

                {/* Validation */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Validation Methodology</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p>
                            We use <strong>SOAP-LOCO</strong> (Smooth Overlap of Atomic Positions — Leave One Cluster Out)
                            splitting to evaluate generalization to unseen chemical families. This approach clusters
                            materials by structural similarity and holds out entire clusters during validation,
                            providing a rigorous test of extrapolation capability.
                        </p>
                    </div>
                </section>

                {/* Performance */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Performance Results</h2>

                    <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-8">
                        <p className="text-green-900 text-sm leading-relaxed">
                            <strong>Governance: APPROVED (6/6 checks passed)</strong> — Ranking (Spearman &gt; 0.5),
                            calibration (90% coverage), KEEP precision (&gt; 85%), false-kill rate (&lt; 2%),
                            and decision-making all verified.
                        </p>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-blue-600 mb-1">0.663</p>
                            <p className="text-sm font-medium text-gray-700">Spearman ρ</p>
                            <p className="text-xs text-gray-500 mt-1">Test set ranking</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-green-600 mb-1">0.030</p>
                            <p className="text-sm font-medium text-gray-700">MAE</p>
                            <p className="text-xs text-gray-500 mt-1">eV/atom (test)</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-indigo-600 mb-1">92.7%</p>
                            <p className="text-sm font-medium text-gray-700">KEEP Precision</p>
                            <p className="text-xs text-gray-500 mt-1">123 / 1,013 KEEP&apos;d</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-teal-600 mb-1">0.0%</p>
                            <p className="text-sm font-medium text-gray-700">False Kill</p>
                            <p className="text-xs text-gray-500 mt-1">Rate</p>
                        </div>
                    </div>

                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-4">
                        <p className="text-amber-900 text-sm">
                            <strong>Known limitation:</strong> LOCO (leave-one-cluster-out) performance degrades
                            significantly (Spearman ≈ 0, coverage 72%). The model is reliable for in-distribution
                            cathodes but should not be trusted for structurally novel polymorphs.
                        </p>
                    </div>

                    <div className="text-sm text-gray-500">
                        <p>
                            <strong>Note:</strong> Primary metrics computed on test split (1,013 materials).
                            Model scope: Transition Metal Oxide cathodes from the Materials Project.
                        </p>
                    </div>
                </section>

                {/* Decision Policy */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Decision Policy</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p className="mb-4">
                            Materials are classified into actionable tiers using conformally-calibrated quantile predictions
                            from the 5-member ensemble:
                        </p>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-4 py-3 text-left font-semibold text-gray-900">Action</th>
                                    <th className="px-4 py-3 text-left font-semibold text-gray-900">Criterion</th>
                                    <th className="px-4 py-3 text-left font-semibold text-gray-900">Interpretation</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                                <tr>
                                    <td className="px-4 py-3 font-medium text-green-700">KEEP</td>
                                    <td className="px-4 py-3 text-gray-600">q90 &lt; 0.05 eV ∧ p_stable &gt; 0.8</td>
                                    <td className="px-4 py-3 text-gray-600">High confidence stable → Prioritize for DFT</td>
                                </tr>
                                <tr>
                                    <td className="px-4 py-3 font-medium text-green-700">KEEP</td>
                                    <td className="px-4 py-3 text-gray-600">q90 &lt; 0.10 eV ∧ p_stable &gt; 0.7</td>
                                    <td className="px-4 py-3 text-gray-600">Likely metastable → Worth DFT validation</td>
                                </tr>
                                <tr>
                                    <td className="px-4 py-3 font-medium text-red-700">KILL</td>
                                    <td className="px-4 py-3 text-gray-600">q10 &gt; 0.10 eV</td>
                                    <td className="px-4 py-3 text-gray-600">Confident unstable → Skip DFT</td>
                                </tr>
                                <tr>
                                    <td className="px-4 py-3 font-medium text-yellow-700">MAYBE</td>
                                    <td className="px-4 py-3 text-gray-600">Otherwise</td>
                                    <td className="px-4 py-3 text-gray-600">Uncertain → Manual review recommended</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* References */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">References</h2>
                    <ol className="space-y-3 text-sm text-gray-600 list-decimal list-inside">
                        <li className="pb-3 border-b border-gray-100">
                            Batatia, I., Benber, P., Chiang, B., et al. (2023).
                            A foundation model for atomistic simulation.
                            <em>arXiv:2401.00096</em>.
                        </li>
                        <li className="pb-3 border-b border-gray-100">
                            Deng, B., Zhong, P., Jun, K., Riebesell, J., Han, K., Bartel, C. J., & Ceder, G. (2023).
                            CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
                            <em>Nature Machine Intelligence</em>, 5(9), 1031–1041.
                        </li>
                        <li className="pb-3 border-b border-gray-100">
                            Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013).
                            Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.
                            <em>APL Materials</em>, 1(1), 011002.
                        </li>
                        <li className="pb-3 border-b border-gray-100">
                            Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
                            Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.
                            <em>Advances in Neural Information Processing Systems</em>, 30.
                        </li>
                        <li className="pb-3 border-b border-gray-100">
                            Bartók, A. P., Kondor, R., & Csányi, G. (2013).
                            On representing chemical environments.
                            <em>Physical Review B</em>, 87(18), 184115.
                        </li>
                        <li>
                            Vovk, V., Gammerman, A., & Shafer, G. (2005).
                            Algorithmic Learning in a Random World.
                            <em>Springer</em>. (Conformal prediction)
                        </li>
                    </ol>
                </section>

                {/* API */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Programmatic Access</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p>
                            CathodeScreen exposes a RESTful API for batch predictions.
                            Upload CIF files and receive JSON responses with predicted E<sub>hull</sub>,
                            uncertainty estimates, and decision recommendations.
                        </p>
                    </div>

                    <div className="bg-slate-800 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                        <div className="text-slate-400 mb-3"># Example API Response</div>
                        <pre className="text-green-400">{`{
  "material_id": "mp-1234567",
  "formula": "LiCoO2",
  "ehull_pred_q50": 0.023,
  "ehull_pred_q10": 0.012,
  "ehull_pred_q90": 0.034,
  "sigma_epistemic": 0.005,
  "sigma_total": 0.008,
  "p_stable": 0.91,
  "decision": "KEEP",
  "conformal_coverage": 0.90
}`}</pre>
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 bg-gray-50">
                <div className="max-w-5xl mx-auto px-6 py-8">
                    <div className="flex flex-col md:flex-row justify-between items-center mb-4">
                        <p className="text-sm text-gray-500 mb-4 md:mb-0">
                            CathodeScreen v1-Li-Cathode - Battery Discovery Platform
                        </p>
                        <SocialLinks />
                    </div>
                    <p className="text-xs text-gray-400 text-center border-t border-gray-200 pt-4">
                        ⚠️ For screening purposes only. DFT validation recommended before synthesis. Model optimized for Li-oxide cathodes.
                    </p>
                </div>
            </footer>
        </div>
    );
}
