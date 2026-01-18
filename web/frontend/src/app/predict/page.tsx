"use client";

import { useState } from "react";
import Link from "next/link";
import API_BASE_URL, { API_KEY } from "@/config";

interface PredictionResult {
    material_id: string;
    pred_ehull: number;
    p_stable: number;
    uncertainty: string;
    action: string;
    confidence_interval: [number, number];
}

export default function Predict() {
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

    const resetForm = () => {
        setFile(null);
        setResult(null);
        setError(null);
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
            {/* Header */}
            <header className="fixed w-full top-4 z-50 flex justify-center pointer-events-none">
                <div className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg rounded-full px-8 py-3 flex items-center gap-12 pointer-events-auto">
                    <Link href="/" className="font-bold text-gray-900 text-lg tracking-tight hover:text-blue-600 hover:scale-125 transition-all duration-200">
                        CathodeScreen
                    </Link>
                    <nav className="flex items-center gap-8 text-sm font-medium text-gray-600">
                        <Link href="/predict" className="text-blue-600 hover:scale-125 transition-all duration-200">Predict</Link>
                        <Link href="/database" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
                        <Link href="/about" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
                    </nav>
                    <div className="pl-8 border-l border-gray-200">
                        <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
                            API <span className="text-[10px]">↗</span>
                        </a>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="pt-32 pb-20 px-6">
                <div className="max-w-4xl mx-auto">
                    {/* Hero */}
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold text-gray-900 mb-4">Predict Material Stability</h1>
                        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                            Upload a CIF file and get instant thermodynamic stability predictions powered by our CHGNet ensemble model.
                        </p>
                    </div>

                    {/* Upload Card */}
                    <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
                        <div className="p-8 md:p-12">
                            {!result ? (
                                <div className="max-w-md mx-auto">
                                    {/* Upload Zone */}
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

                                    {/* Submit Button */}
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
                                                Running CHGNet Inference...
                                            </span>
                                        ) : "Predict Stability"}
                                    </button>

                                    {/* Error */}
                                    {error && (
                                        <div className="mt-6 p-4 bg-red-50 border border-red-100 rounded-lg text-red-600 text-sm text-center">
                                            {error}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                /* Results Display */
                                <div>
                                    <div className="flex items-center justify-between mb-8">
                                        <h3 className="text-2xl font-bold text-gray-900">Prediction Result</h3>
                                        <div className="flex items-center gap-4">
                                            <span className={`px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wider ${result.action === "DFT" || result.action === "KEEP"
                                                ? "bg-green-100 text-green-700"
                                                : result.action === "HOLD" || result.action === "MAYBE"
                                                    ? "bg-yellow-100 text-yellow-700"
                                                    : "bg-red-100 text-red-600"
                                                }`}>
                                                {result.action}
                                            </span>
                                            <button
                                                onClick={resetForm}
                                                className="text-sm text-gray-500 hover:text-blue-600 underline"
                                            >
                                                New Prediction
                                            </button>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                                        <div className="bg-gray-50 rounded-xl p-6">
                                            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">E<sub>hull</sub></p>
                                            <p className="text-3xl font-bold text-gray-900">{result.pred_ehull.toFixed(3)}</p>
                                            <p className="text-sm text-gray-400 mt-1">eV/atom</p>
                                        </div>
                                        <div className="bg-gray-50 rounded-xl p-6">
                                            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">P(Stable)</p>
                                            <p className="text-3xl font-bold text-gray-900">{(result.p_stable * 100).toFixed(0)}%</p>
                                            <p className="text-sm text-gray-400 mt-1">Confidence</p>
                                        </div>
                                        <div className="bg-gray-50 rounded-xl p-6">
                                            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Uncertainty</p>
                                            <p className={`text-3xl font-bold ${result.uncertainty === "Low" ? "text-green-600" : result.uncertainty === "Medium" ? "text-yellow-600" : "text-red-600"}`}>
                                                {result.uncertainty}
                                            </p>
                                            <p className="text-sm text-gray-400 mt-1">Epistemic</p>
                                        </div>
                                        <div className="bg-gray-50 rounded-xl p-6">
                                            <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">95% CI</p>
                                            <p className="text-xl font-bold text-gray-900 mt-2">
                                                [{result.confidence_interval[0].toFixed(3)}, {result.confidence_interval[1].toFixed(3)}]
                                            </p>
                                            <p className="text-sm text-gray-400 mt-1">eV/atom</p>
                                        </div>
                                    </div>

                                    {/* OOD / High Uncertainty Warning */}
                                    {result.uncertainty === "High" && (
                                        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
                                            <div className="text-red-500 mt-0.5">
                                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                                </svg>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold text-red-800">High Uncertainty - Treat with Caution</h4>
                                                <p className="text-sm text-red-600">This prediction has high epistemic uncertainty, indicating the material may be outside the model&apos;s training domain. DFT validation is strongly recommended.</p>
                                            </div>
                                        </div>
                                    )}

                                    {/* Medium Uncertainty Warning */}
                                    {result.uncertainty === "Medium" && (
                                        <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-3">
                                            <div className="text-amber-500 mt-0.5">
                                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                                </svg>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold text-amber-800">Moderate Uncertainty</h4>
                                                <p className="text-sm text-amber-600">Ensemble models show some disagreement. Consider the confidence interval when interpreting results.</p>
                                            </div>
                                        </div>
                                    )}

                                    {/* Interpretation */}
                                    <div className="p-6 bg-blue-50 rounded-xl border border-blue-100">
                                        <h4 className="font-semibold text-blue-900 mb-2">Interpretation</h4>
                                        <p className="text-sm text-blue-700">
                                            {result.action === "KEEP" || result.action === "DFT" ? (
                                                <>This material is predicted to be thermodynamically stable. <strong>Recommended for DFT validation.</strong></>
                                            ) : result.action === "MAYBE" || result.action === "HOLD" ? (
                                                <>This material has borderline stability. Consider experimental validation or additional DFT studies.</>
                                            ) : (
                                                <>This material is predicted to be unstable. It may not be a viable candidate for synthesis.</>
                                            )}
                                        </p>
                                        <p className="text-xs text-blue-500 mt-3">
                                            95% Confidence Interval: [{result.confidence_interval[0].toFixed(3)}, {result.confidence_interval[1].toFixed(3)}] eV/atom
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Info Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
                        <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4 text-blue-600 font-bold">1</div>
                            <h3 className="font-semibold text-gray-900 mb-2">Upload Structure</h3>
                            <p className="text-sm text-gray-600">Drag & drop your CIF file. Our model parses 3D crystal structures instantly.</p>
                        </div>
                        <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center mb-4 text-indigo-600 font-bold">2</div>
                            <h3 className="font-semibold text-gray-900 mb-2">CHGNet Inference</h3>
                            <p className="text-sm text-gray-600">5-member CHGNet ensemble predicts E<sub>hull</sub> with calibrated uncertainty.</p>
                        </div>
                        <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600 font-bold">3</div>
                            <h3 className="font-semibold text-gray-900 mb-2">Get Recommendation</h3>
                            <p className="text-sm text-gray-600">Receive KEEP, MAYBE, or KILL recommendations to prioritize your research.</p>
                        </div>
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 bg-white">
                <div className="max-w-5xl mx-auto px-6 py-6">
                    <div className="flex justify-between items-center text-sm text-gray-500 mb-3">
                        <p>CathodeScreen · CHGNet v1-Li-Cathode</p>
                        <div className="flex gap-4">
                            <a href="https://github.com/ErenAri" target="_blank" className="hover:text-gray-900">GitHub</a>
                            <a href={`${API_BASE_URL}/docs`} target="_blank" className="hover:text-gray-900">API</a>
                            <Link href="/about" className="hover:text-gray-900">About</Link>
                        </div>
                    </div>
                    <p className="text-xs text-gray-400 text-center border-t border-gray-100 pt-3">
                        ⚠️ For screening purposes only. DFT validation is recommended before synthesis. Model is optimized for Li-containing oxide cathode materials.
                    </p>
                </div>
            </footer>
        </div>
    );
}
