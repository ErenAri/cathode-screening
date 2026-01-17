"use client";

import { useState } from "react";
import API_BASE_URL, { API_KEY } from "@/config";
import { SocialLinks } from "@/components/SocialLinks";

interface PredictionResult {
  material_id: string;
  pred_ehull: number;
  p_stable: number;
  uncertainty: string;
  action: string;
  confidence_interval: [number, number];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("cif_file", file);

      // Use configured API URL
      const headers: HeadersInit = {};
      if (API_KEY) {
        headers["X-API-Key"] = API_KEY;
      }

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers,
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.prediction);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch {
      setError("Failed to connect to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white font-sans text-gray-900">
      {/* Header */}
      <header className="fixed w-full bg-white/80 backdrop-blur-md z-50 border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold tracking-tight text-gray-900">CathodeScreen</h1>

          <nav className="flex gap-8 text-sm font-medium">
            <a href="#predict" className="text-blue-600">Predict</a>
            <a href="#results" className="text-gray-600 hover:text-gray-900">Results</a>
            <a href="/database" className="text-gray-600 hover:text-gray-900">Database</a>
            <a href="/about" className="text-gray-600 hover:text-gray-900">About</a>
            <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-gray-600 hover:text-gray-900">API</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 max-w-6xl mx-auto text-center">
        <div className="inline-block px-4 py-1.5 bg-blue-50 text-blue-700 rounded-full text-sm font-medium mb-6">
          Enterprise AI Screening
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-6 text-gray-900">
          Accelerate Battery Material <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
            Discovery with AI
          </span>
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-10 leading-relaxed">
          Instantly predict thermodynamic stability of Li-cathode materials using our state-of-the-art
          CHGNet ensemble model. Screen with 6.6× enrichment and &lt;0.3% false-discard rate.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto mt-16 text-left">
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4 text-blue-600 font-bold">1</div>
            <h3 className="font-semibold text-lg mb-2">Upload Structure</h3>
            <p className="text-gray-600 text-sm">Drag & drop your CIF file. Our model parses 3D crystal structures instantly.</p>
          </div>
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center mb-4 text-indigo-600 font-bold">2</div>
            <h3 className="font-semibold text-lg mb-2">AI Inference</h3>
            <p className="text-gray-600 text-sm">5-member Graph Neural Network ensemble predicts E<sub>hull</sub> and Uncertainty.</p>
          </div>
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600 font-bold">3</div>
            <h3 className="font-semibold text-lg mb-2">Get Actionable Insight</h3>
            <p className="text-gray-600 text-sm">Receive immediate "DFT", "HOLD", or "SKIP" recommendations.</p>
          </div>
        </div>
      </section>

      {/* Main Tool */}
      <section id="predict" className="py-20 bg-gray-50 border-y border-gray-200">
        <div className="max-w-4xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Try it now</h2>
            <p className="text-gray-600">Upload a `.cif` file to see the model in action.</p>
          </div>

          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div className="p-8 md:p-12">
              <div className="max-w-md mx-auto">
                <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 bg-gray-50 hover:bg-white hover:border-blue-400 transition-colors text-center group cursor-pointer relative">
                  <input
                    type="file"
                    accept=".cif"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="mb-4 text-gray-400 group-hover:text-blue-500 transition-colors">
                    <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="font-medium text-gray-900">
                    {file ? file.name : "Click to upload CIF"}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    {file ? "Ready to predict" : "or drag and drop"}
                  </p>
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!file || loading}
                  className={`mt-6 w-full py-3 px-6 rounded-xl font-semibold shadow-lg transition-all transform hover:-translate-y-0.5 ${file && !loading
                    ? "bg-blue-600 text-white hover:bg-blue-700 shadow-blue-200"
                    : "bg-gray-200 text-gray-400 cursor-not-allowed shadow-none"
                    }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Running Inference...
                    </span>
                  ) : "Predict Stability"}
                </button>
              </div>

              {/* Error */}
              {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-100 rounded-lg text-red-600 text-sm text-center">
                  {error}
                </div>
              )}

              {/* Result */}
              {result && (
                <div className="mt-10 border-t border-gray-100 pt-10">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-bold text-gray-900">Prediction Result</h3>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${result.action === "DFT"
                      ? "bg-green-100 text-green-700"
                      : result.action === "HOLD"
                        ? "bg-yellow-100 text-yellow-700"
                        : "bg-gray-100 text-gray-600"
                      }`}>
                      {result.action}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="bg-gray-50 rounded-xl p-4">
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">E<sub>hull</sub></p>
                      <p className="text-2xl font-bold text-gray-900">{result.pred_ehull.toFixed(3)}</p>
                      <p className="text-xs text-gray-400">eV/atom</p>
                    </div>
                    <div className="bg-gray-50 rounded-xl p-4">
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">P(Stable)</p>
                      <p className="text-2xl font-bold text-gray-900">{(result.p_stable * 100).toFixed(0)}%</p>
                      <p className="text-xs text-gray-400">Confidence</p>
                    </div>
                    <div className="bg-gray-50 rounded-xl p-4">
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Uncertainty</p>
                      <p className={`text-2xl font-bold ${result.uncertainty === "Low" ? "text-green-600" : "text-yellow-600"
                        }`}>{result.uncertainty}</p>
                      <p className="text-xs text-gray-400">Epistemic</p>
                    </div>
                    <div className="bg-gray-50 rounded-xl p-4">
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Interval</p>
                      <p className="text-lg font-bold text-gray-900 mt-1">
                        {result.confidence_interval[0].toFixed(2)} - {result.confidence_interval[1].toFixed(2)}
                      </p>
                      <p className="text-xs text-gray-400">95% CI</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Results / Methodology Section */}
      <section id="results" className="py-20 px-6 max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-6">Built on Rigorous Science</h2>
            <p className="text-lg text-gray-600 mb-6">
              We leverage a deep ensemble of CHGNet (Crystal Hamiltonian Graph Neural Network) models, fine-tuned on 11,377 Li-cathode materials.
            </p>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">✓</div>
                <div>
                  <h4 className="font-semibold text-gray-900">SOAP-LOCO Validation</h4>
                  <p className="text-sm text-gray-600">Validated on clustered hold-out sets to ensure generalization to new chemistries.</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">✓</div>
                <div>
                  <h4 className="font-semibold text-gray-900">Uncertainty Quantification</h4>
                  <p className="text-sm text-gray-600">5-model ensemble provides epistemic uncertainty, helping you trust high-stakes predictions.</p>
                </div>
              </li>
            </ul>
            <a href="/about" className="text-blue-600 font-semibold hover:text-blue-700 flex items-center gap-2 group">
              Read full methodology
              <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" /></svg>
            </a>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 text-center">
              <p className="text-4xl font-extrabold text-blue-600 mb-2">1.64×</p>
              <p className="font-semibold text-gray-900">Discovery Factor</p>
              <p className="text-xs text-gray-500 mt-2">More effective than random search (DAF@10)</p>
            </div>
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 text-center">
              <p className="text-4xl font-extrabold text-green-600 mb-2">+86%</p>
              <p className="font-semibold text-gray-900">Efficiency</p>
              <p className="text-xs text-gray-500 mt-2">Active Learning gain vs. Baseline</p>
            </div>
            <div className="col-span-2 bg-gray-900 p-8 rounded-2xl text-white text-center">
              <p className="text-3xl font-bold mb-2">17,227</p>
              <p className="text-gray-400">Materials Learned</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 py-12">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-500 mb-4">CathodeScreen · Enterprise Platform · 2026</p>
          <div className="flex justify-center mb-6">
            <SocialLinks />
          </div>
          <div className="flex justify-center gap-6 text-sm text-gray-600">
            <a href="/about" className="hover:text-blue-600">About</a>
            <a href="/privacy" className="hover:text-blue-600">Privacy</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
