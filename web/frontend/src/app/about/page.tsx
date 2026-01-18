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
                            We employ <strong>CHGNet</strong> (Crystal Hamiltonian Graph Neural Network),
                            a state-of-the-art universal neural network potential pretrained on the Materials Project
                            relaxation trajectories (Deng et al., 2023). CHGNet represents crystal structures as
                            atom-bond graphs with charge-informed embeddings, enabling accurate prediction of
                            energy above the convex hull (E<sub>hull</sub>).
                        </p>
                    </div>

                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 mb-6">
                        <h3 className="font-semibold text-gray-900 mb-4">Model Configuration (v1-Li-Cathode)</h3>
                        <div className="grid md:grid-cols-2 gap-6 text-sm text-gray-700">
                            <div>
                                <h4 className="font-medium text-gray-900 mb-2">Architecture</h4>
                                <ul className="space-y-1">
                                    <li>• CHGNet v0.3.0 (412,525 parameters)</li>
                                    <li>• 5-member deep ensemble</li>
                                    <li>• Independent random seeds (42, 123, 456, 789, 1024)</li>
                                    <li>• Fine-tuned prediction head for E<sub>hull</sub></li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-medium text-gray-900 mb-2">Training Details</h4>
                                <ul className="space-y-1">
                                    <li>• Dataset: 11,377 Li-O-TM cathodes</li>
                                    <li>• Splitting: SOAP-LOCO (Leave-One-Cluster-Out)</li>
                                    <li>• Target: E<sub>hull</sub> in eV/atom</li>
                                    <li>• Optimizer: AdamW, cosine annealing</li>
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
                            Following Lakshminarayanan et al. (2017), we implement <strong>Deep Ensembles</strong> for
                            epistemic uncertainty estimation. The ensemble mean provides the point prediction,
                            while ensemble variance captures model uncertainty:
                        </p>
                    </div>

                    <div className="bg-slate-800 text-slate-100 rounded-lg p-6 font-mono text-sm mb-6">
                        <p className="mb-2">μ = (1/M) Σ<sub>m</sub> μ<sub>m</sub>(x)</p>
                        <p>σ² = (1/M) Σ<sub>m</sub> (μ<sub>m</sub>(x)² − μ²)</p>
                    </div>

                    <p className="text-gray-600">
                        Calibrated confidence intervals are obtained via conformal prediction,
                        ensuring valid coverage guarantees under distribution shift.
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

                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
                        <p className="text-blue-900 text-sm leading-relaxed">
                            <strong>Key Finding:</strong> On SOAP-LOCO holdout chemistries, the v1-Li-Cathode ensemble
                            achieves a <strong>6.6× enrichment factor</strong> at the 1% threshold (E<sub>hull</sub> &lt; 10 meV),
                            recovering 55% of ultra-stable materials within the top-100 ranked candidates while
                            maintaining a <strong>&lt;0.3% false kill rate</strong>.
                        </p>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-blue-600 mb-1">6.66×</p>
                            <p className="text-sm font-medium text-gray-700">EF@1%</p>
                            <p className="text-xs text-gray-500 mt-1">Enrichment Factor</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-green-600 mb-1">0.032</p>
                            <p className="text-sm font-medium text-gray-700">MAE</p>
                            <p className="text-xs text-gray-500 mt-1">eV/atom</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-indigo-600 mb-1">55%</p>
                            <p className="text-sm font-medium text-gray-700">Recall@100</p>
                            <p className="text-xs text-gray-500 mt-1">at 10 meV</p>
                        </div>
                        <div className="bg-white border border-gray-200 rounded-lg p-5 text-center">
                            <p className="text-3xl font-bold text-teal-600 mb-1">&lt;0.3%</p>
                            <p className="text-sm font-medium text-gray-700">False Kill</p>
                            <p className="text-xs text-gray-500 mt-1">Rate</p>
                        </div>
                    </div>

                    <div className="text-sm text-gray-500">
                        <p>
                            <strong>Note:</strong> Metrics computed on SOAP-LOCO test split (764 materials).
                            Model scope limited to Li-O-TM ternary/quaternary cathodes.
                        </p>
                    </div>
                </section>

                {/* Decision Policy */}
                <section className="mb-16">
                    <h2 className="text-2xl font-bold text-gray-900 mb-4">Decision Policy</h2>
                    <div className="prose prose-gray max-w-none text-gray-600 mb-6">
                        <p className="mb-4">
                            Materials are classified into three tiers based on predicted E<sub>hull</sub> and uncertainty:
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
                                    <td className="px-4 py-3 text-gray-600">μ &lt; 0.05 eV ∧ σ &lt; 0.02</td>
                                    <td className="px-4 py-3 text-gray-600">High confidence stable → Prioritize for DFT</td>
                                </tr>
                                <tr>
                                    <td className="px-4 py-3 font-medium text-yellow-700">MAYBE</td>
                                    <td className="px-4 py-3 text-gray-600">0.05 ≤ μ ≤ 0.15 ∨ σ &gt; 0.02</td>
                                    <td className="px-4 py-3 text-gray-600">Uncertain → Manual review recommended</td>
                                </tr>
                                <tr>
                                    <td className="px-4 py-3 font-medium text-red-700">KILL</td>
                                    <td className="px-4 py-3 text-gray-600">μ &gt; 0.15 eV</td>
                                    <td className="px-4 py-3 text-gray-600">Confident unstable → Skip DFT</td>
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
                        <li>
                            Bartók, A. P., Kondor, R., & Csányi, G. (2013).
                            On representing chemical environments.
                            <em>Physical Review B</em>, 87(18), 184115.
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
  "ehull_pred": 0.023,
  "uncertainty": 0.008,
  "ci_lower": 0.012,
  "ci_upper": 0.034,
  "decision": "KEEP",
  "confidence": 0.92
}`}</pre>
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 bg-gray-50">
                <div className="max-w-5xl mx-auto px-6 py-8 flex flex-col md:flex-row justify-between items-center">
                    <p className="text-sm text-gray-500 mb-4 md:mb-0">
                        CathodeScreen v1-Li-Cathode · Enterprise Material Discovery Platform
                    </p>
                    <SocialLinks />
                </div>
            </footer>
        </div>
    );
}
