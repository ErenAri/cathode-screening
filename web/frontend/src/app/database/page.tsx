"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import API_BASE_URL, { API_KEY } from "@/config";
import { SocialLinks } from "@/components/SocialLinks";

interface MaterialPrediction {
    material_id: string;
    formula: string;
    pred_ehull: number;
    p_stable: number;
    uncertainty: string;
    action: string;
}

export default function Database() {
    const [search, setSearch] = useState("");
    const [data, setData] = useState<MaterialPrediction[]>([]);
    const [filtered, setFiltered] = useState<MaterialPrediction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [page, setPage] = useState(1);
    const perPage = 50;

    // Fetch data from API
    useEffect(() => {
        const headers: HeadersInit = {};
        if (API_KEY) {
            headers["X-API-Key"] = API_KEY;
        }

        fetch(`${API_BASE_URL}/database`, { headers })
            .then((res) => res.json())
            .then((result) => {
                if (result.success) {
                    setData(result.data);
                    setFiltered(result.data);
                } else {
                    setError(result.error || "Failed to load data");
                }
                setLoading(false);
            })
            .catch(() => {
                setError("Failed to connect to server");
                setLoading(false);
            });
    }, []);

    // Filter by search
    useEffect(() => {
        const q = search.toLowerCase();
        const results = data.filter(
            (m) =>
                m.material_id.toLowerCase().includes(q) ||
                m.formula.toLowerCase().includes(q)
        );
        setFiltered(results);
        setPage(1);
    }, [search, data]);

    const stats = {
        total: data.length,
        dft: data.filter((m) => m.action === "DFT").length,
        hold: data.filter((m) => m.action === "HOLD").length,
        skip: data.filter((m) => m.action === "SKIP").length,
    };

    // Download CSV
    const downloadCSV = () => {
        const headers = ["material_id", "formula", "pred_ehull", "p_stable", "uncertainty", "action"];
        const rows = filtered.map((m) =>
            [m.material_id, m.formula, m.pred_ehull, m.p_stable, m.uncertainty, m.action].join(",")
        );
        const csv = [headers.join(","), ...rows].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "cathodescreen_predictions.csv";
        a.click();
    };

    // Pagination
    const totalPages = Math.ceil(filtered.length / perPage);
    const displayed = filtered.slice((page - 1) * perPage, page * perPage);

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
                        <Link href="/database" className="text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
                        <Link href="/about" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
                    </nav>
                    <div className="pl-8 border-l border-gray-200">
                        <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
                            API <span className="text-[10px]">↗</span>
                        </a>
                    </div>
                </div>
            </header>

            {/* Main */}
            <main className="max-w-5xl mx-auto px-6 py-10 pt-32">
                <div className="flex justify-between items-start mb-6">
                    <div>
                        <h1 className="text-2xl font-semibold text-gray-900 mb-2">
                            Pre-computed Predictions
                        </h1>
                        <p className="text-gray-600">
                            Browse stability predictions for cathode materials from Materials Project.
                        </p>
                    </div>
                    <button
                        onClick={downloadCSV}
                        disabled={loading || filtered.length === 0}
                        className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                        Download CSV
                    </button>
                </div>

                {loading ? (
                    <div className="text-center py-12 text-gray-500">Loading...</div>
                ) : error ? (
                    <div className="text-center py-12 text-red-500">{error}</div>
                ) : (
                    <>
                        {/* Stats */}
                        <div className="grid grid-cols-4 gap-4 mb-8">
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-gray-900">{stats.total.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Total Materials</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-green-600">{stats.dft.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Recommended for DFT</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-yellow-600">{stats.hold.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Hold for Review</p>
                            </div>
                            <div className="border border-gray-200 rounded-lg p-4">
                                <p className="text-2xl font-semibold text-gray-400">{stats.skip.toLocaleString()}</p>
                                <p className="text-sm text-gray-500">Skip</p>
                            </div>
                        </div>

                        {/* Search */}
                        <div className="mb-4">
                            <input
                                type="text"
                                placeholder="Search by material ID or formula..."
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                className="w-full max-w-md px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                        </div>

                        {/* Table */}
                        <div className="border border-gray-200 rounded-lg overflow-hidden">
                            <table className="w-full text-sm">
                                <thead className="bg-gray-50 border-b border-gray-200">
                                    <tr>
                                        <th className="px-4 py-3 text-left font-medium text-gray-600">Material ID</th>
                                        <th className="px-4 py-3 text-left font-medium text-gray-600">Formula</th>
                                        <th className="px-4 py-3 text-right font-medium text-gray-600">E<sub>hull</sub> (eV)</th>
                                        <th className="px-4 py-3 text-right font-medium text-gray-600">P(Stable)</th>
                                        <th className="px-4 py-3 text-center font-medium text-gray-600">Uncertainty</th>
                                        <th className="px-4 py-3 text-center font-medium text-gray-600">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {displayed.map((m, i) => (
                                        <tr key={m.material_id} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                                            <td className="px-4 py-3 font-mono text-blue-700 font-medium">
                                                <a
                                                    href={`https://materialsproject.org/materials/${m.material_id}`}
                                                    target="_blank"
                                                    className="hover:underline"
                                                >
                                                    {m.material_id}
                                                </a>
                                            </td>
                                            <td className="px-4 py-3 text-gray-900 font-medium">{m.formula || "-"}</td>
                                            <td className="px-4 py-3 text-right font-mono text-gray-800">{m.pred_ehull.toFixed(3)}</td>
                                            <td className="px-4 py-3 text-right text-gray-800">{(m.p_stable * 100).toFixed(0)}%</td>
                                            <td className="px-4 py-3 text-center">
                                                <span className={`text-xs ${m.uncertainty === "Low" ? "text-green-600" :
                                                    m.uncertainty === "Medium" ? "text-yellow-600" : "text-red-600"
                                                    }`}>
                                                    {m.uncertainty}
                                                </span>
                                            </td>
                                            <td className="px-4 py-3 text-center">
                                                <span className={`px-2 py-1 rounded text-xs font-medium ${m.action === "DFT" ? "bg-green-100 text-green-800" :
                                                    m.action === "HOLD" ? "bg-yellow-100 text-yellow-800" :
                                                        "bg-gray-100 text-gray-600"
                                                    }`}>
                                                    {m.action}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Pagination */}
                        <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
                            <p>
                                Showing {(page - 1) * perPage + 1}-{Math.min(page * perPage, filtered.length)} of {filtered.length.toLocaleString()} materials
                            </p>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setPage(p => Math.max(1, p - 1))}
                                    disabled={page === 1}
                                    className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                >
                                    Previous
                                </button>
                                <span className="px-3 py-1">
                                    Page {page} of {totalPages}
                                </span>
                                <button
                                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    disabled={page === totalPages}
                                    className="px-3 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50"
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    </>
                )}
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-200 mt-16">
                <div className="max-w-5xl mx-auto px-6 py-6">
                    <div className="flex justify-between text-sm text-gray-500 mb-3">
                        <p>CathodeScreen · Enterprise Platform</p>
                        <div className="flex gap-4">
                            <a href="https://github.com/ErenAri" target="_blank" className="hover:text-gray-900">GitHub</a>
                            <a href={`${API_BASE_URL}/docs`} target="_blank" className="hover:text-gray-900">API</a>
                            <Link href="/about" className="hover:text-gray-900">About</Link>
                        </div>
                    </div>
                    <p className="text-xs text-gray-400 text-center border-t border-gray-100 pt-3">
                        ⚠️ For screening purposes only. DFT validation recommended before synthesis. Model optimized for Li-oxide cathodes.
                    </p>
                </div>
            </footer>
        </div>
    );
}
