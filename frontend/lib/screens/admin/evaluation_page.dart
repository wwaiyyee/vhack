import 'package:flutter/material.dart';
import 'mock_data.dart';

class EvaluationPage extends StatelessWidget {
  const EvaluationPage({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          const Row(
            children: [
              Icon(Icons.analytics_outlined, color: Color(0xFF6366F1), size: 22),
              SizedBox(width: 10),
              Text('Model Evaluation', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B))),
            ],
          ),
          const SizedBox(height: 8),
          const Text('Compare the candidate model against the currently deployed model on the holdout validation set.',
            style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
          ),
          const SizedBox(height: 28),

          // Overview Cards
          Row(
            children: [
              _modelInfoCard('Current Deployed', mockCurrentModelVersion, const Color(0xFF64748B), true),
              const SizedBox(width: 24),
              const Icon(Icons.compare_arrows_rounded, color: Color(0xFF94A3B8)),
              const SizedBox(width: 24),
              _modelInfoCard('Candidate', mockCandidateModelVersion, const Color(0xFF6366F1), false),
            ],
          ),
          const SizedBox(height: 32),

          // Metrics Table
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Validation Metrics', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
                const SizedBox(height: 16),
                Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFE2E8F0)),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Column(
                      children: [
                        // Header
                        Container(
                          color: const Color(0xFFF8FAFC),
                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                          child: const Row(
                            children: [
                              Expanded(flex: 3, child: Text('Metric', style: _headerStyle)),
                              Expanded(flex: 2, child: Text('Current', style: _headerStyle)),
                              Expanded(flex: 2, child: Text('Candidate', style: _headerStyle)),
                              Expanded(flex: 2, child: Text('Delta', style: _headerStyle)),
                            ],
                          ),
                        ),
                        // Rows
                        ...mockMetrics.asMap().entries.map((entry) {
                          final idx = entry.key;
                          final m = entry.value;
                          final isLast = idx == mockMetrics.length - 1;
                          
                          // Determine if candidate is better.
                          // For Acuuracy, Precision, Recall, F1 higher is better.
                          // For FPR, FNR lower is better.
                          final lowerIsBetter = m.name.contains('Rate');
                          final diff = m.candidateValue - m.currentValue;
                          final isBetter = lowerIsBetter ? (diff < 0) : (diff > 0);
                          final isNeutral = diff == 0;
                          
                          final diffColor = isNeutral ? const Color(0xFF64748B) : (isBetter ? const Color(0xFF22C55E) : const Color(0xFFEF4444));
                          final diffIcon = diff > 0 ? Icons.arrow_upward_rounded : (diff < 0 ? Icons.arrow_downward_rounded : Icons.remove);
                          final diffText = '${(diff.abs() * 100).toStringAsFixed(1)}%';

                          return Container(
                            decoration: BoxDecoration(
                              border: isLast ? null : const Border(bottom: BorderSide(color: Color(0xFFF1F5F9))),
                            ),
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
                            child: Row(
                              children: [
                                Expanded(flex: 3, child: Text(m.name, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF334155)))),
                                Expanded(flex: 2, child: Text('${(m.currentValue * 100).toStringAsFixed(1)}%', style: const TextStyle(fontSize: 14, color: Color(0xFF64748B)))),
                                Expanded(flex: 2, child: Text('${(m.candidateValue * 100).toStringAsFixed(1)}%', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B)))),
                                Expanded(
                                  flex: 2, 
                                  child: Row(
                                    children: [
                                      Icon(diffIcon, size: 14, color: diffColor),
                                      const SizedBox(width: 4),
                                      Text(diffText, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: diffColor)),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          );
                        }),
                      ],
                    ),
                  ),
                ),
                
                const SizedBox(height: 24),
                // Summary conclusion
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF22C55E).withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFF22C55E).withValues(alpha: 0.2)),
                  ),
                  child: const Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.verified_rounded, color: Color(0xFF22C55E), size: 24),
                      SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Candidate Outperforms Current Model', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF166534))),
                            SizedBox(height: 4),
                            Text(
                              'The candidate model shows consistent improvements across all major metrics, particularly reducing False Negative Rate by 3.1%. It is recommended for deployment.',
                              style: TextStyle(fontSize: 13, color: Color(0xFF14532D), height: 1.4),
                            ),
                          ],
                        ),
                      )
                    ],
                  ),
                )
              ],
            ),
          )
        ],
      ),
    );
  }

  Widget _modelInfoCard(String title, String version, Color color, bool isCurrent) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: isCurrent ? const Color(0xFFE2E8F0) : color.withValues(alpha: 0.3), width: isCurrent ? 1 : 2),
          boxShadow: isCurrent ? [] : [
            BoxShadow(color: color.withValues(alpha: 0.1), blurRadius: 10, offset: const Offset(0, 4)),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF64748B))),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.inventory_2_outlined, size: 20, color: color),
                const SizedBox(width: 8),
                Text(version, style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: color, fontFamily: 'monospace')),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
      ),
      child: child,
    );
  }

  static const _headerStyle = TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Color(0xFF94A3B8), letterSpacing: 0.5);
}
