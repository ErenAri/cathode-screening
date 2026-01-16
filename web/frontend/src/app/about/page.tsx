import Link from "next/link";
import { SocialLinks } from "@/components/SocialLinks";
import API_BASE_URL from "@/config";

export default function About() {
    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="border-b border-gray-200">
                <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="text-xl font-semibold text-gray-900">
                        CathodeScreen
                    </Link>
                    <nav className="flex gap-6 text-sm">
                        <Link href="/" className="text-gray-600 hover:text-gray-900">Predict</Link>
                        <Link href="/database" className="text-gray-600 hover:text-gray-900">Database</Link>
                        <Link href="/about" className="text-blue-600 font-medium">About</Link>
                        <a href={`${API_BASE_URL}/docs`} className="text-gray-600 hover:text-gray-900">API</a>
                    </nav>
                </div>
            </header>

            {/* Main */}
            <main className="max-w-3xl mx-auto px-6 py-12">
                <h1 className="text-3xl font-semibold text-gray-900 mb-6">About CathodeScreen</h1>

                <div className="prose prose-gray max-w-none">
                    <p className="text-lg text-gray-600 mb-8">
                        CathodeScreen is an open-source machine learning tool for predicting the thermodynamic
                        stability of lithium-ion battery cathode materials.
                    </p>

                    <h2 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Method</h2>
                    <p className="text-gray-600 mb-4">
                        We use a <strong>Crystal Graph Convolutional Neural Network (CGCNN)</strong> to predict
                        the energy above the convex hull (E<sub>hull</sub>), the primary indicator of thermodynamic stability.
                    </p>

                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 my-6">
                        <h3 className="font-semibold text-gray-900 mb-2">Model Architecture</h3>
                        <ul className="text-sm text-gray-600 space-y-1">
                            <li>• 92-dimensional CGCNN atom embeddings</li>
                            <li>• 4-layer message passing with 128 hidden units</li>
                            <li>• Multi-head attention pooling (4 heads)</li>
                            <li>• 5-member deep ensemble for uncertainty quantification</li>
                        </ul>
                    </div>

                    <h2 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Training Data</h2>
                    <p className="text-gray-600 mb-4">
                        The model was trained on <strong>17,227 cathode materials</strong> from the Materials Project
                        database, using SOAP-LOCO (Smooth Overlap of Atomic Positions - Leave One Cluster Out)
                        splitting to ensure rigorous out-of-distribution evaluation.
                    </p>

                    <h2 className="text-xl font-semibold text-gray-900 mt-8 mb-4">Performance</h2>
                    <div className="grid grid-cols-2 gap-4 my-6">
                        <div className="border border-gray-200 rounded-lg p-4 text-center">
                            <p className="text-3xl font-bold text-blue-600">1.64×</p>
                            <p className="text-sm text-gray-500">DAF@10</p>
                            <p className="text-xs text-gray-400 mt-1">Discovery Acceleration Factor</p>
                        </div>
                        <div className="border border-gray-200 rounded-lg p-4 text-center">
                            <p className="text-3xl font-bold text-green-600">+86%</p>
                            <p className="text-sm text-gray-500">Active Learning</p>
                            <p className="text-xs text-gray-400 mt-1">vs random sampling</p>
                        </div>
                    </div>



                    <h2 className="text-xl font-semibold text-gray-900 mt-8 mb-4">References</h2>
                    <ol className="text-sm text-gray-600 space-y-2 list-decimal list-inside">
                        <li>Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks. <em>Phys. Rev. Lett.</em></li>
                        <li>Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. <em>APL Mater.</em></li>
                        <li>Lakshminarayanan, B., et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. <em>NeurIPS</em>.</li>
                        <li>Bartók, A. P., et al. (2013). On representing chemical environments. <em>Phys. Rev. B</em></li>
                    </ol>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 mt-16">
                <div className="max-w-4xl mx-auto px-6 py-6 flex justify-between text-sm text-gray-500 items-center">
                    <p>CathodeScreen · Enterprise Platform</p>
                    <SocialLinks />
                </div>
            </footer>
        </div>
    );
}
