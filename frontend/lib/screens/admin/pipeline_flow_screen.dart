import 'package:flutter/material.dart';
import 'mock_data.dart';

class PipelineFlowScreen extends StatefulWidget {
  const PipelineFlowScreen({super.key});

  @override
  State<PipelineFlowScreen> createState() => _PipelineFlowScreenState();
}

class _PipelineFlowScreenState extends State<PipelineFlowScreen> {
  // Navigation State
  int _currentStage = 1; // 1: Retraining, 2: Evaluation, 3: Deployment

  // Stage 1 State
  String? _selectedModel;
  bool _isTraining = false;
  int _currentStepIndex = -1;

  // Stage 3 State
  bool _isDeploying = false;
  bool _isDeployed = false;
  bool _isRejected = false;

  void _startTraining() async {
    if (_selectedModel == null) return;
    
    setState(() {
      _isTraining = true;
      _currentStepIndex = 0;
    });

    for (int i = 0; i < mockTrainingSteps.length; i++) {
      if (!mounted) return;
      setState(() => _currentStepIndex = i);
      
      int ms = 5000; // default for last steps
      if (i == 0) ms = 10000; // Loading: 10s
      else if (i == 1) ms = 12300; // Preprocess: 12.3s
      else if (i >= 2 && i < 12) ms = 18000; // 10 Epochs: 18s each = 3 minutes total
      
      await Future.delayed(Duration(milliseconds: ms));
    }

    if (mounted) {
      setState(() {
        _isTraining = false;
        _currentStage = 2; // Auto-advance to Evaluation
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Retraining completed. Candidate ready for evaluation.'), backgroundColor: Color(0xFF22C55E)),
      );
    }
  }

  void _deploy() async {
    setState(() { _isDeploying = true; });
    await Future.delayed(const Duration(seconds: 2));
    if (mounted) {
      setState(() {
        _isDeploying = false;
        _isDeployed = true;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model deployed successfully!'), backgroundColor: Color(0xFF22C55E)),
      );
    }
  }

  void _reject() {
    setState(() { _isRejected = true; });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          const Row(
            children: [
              Icon(Icons.hub_outlined, color: Color(0xFF6366F1), size: 22),
              SizedBox(width: 10),
              Text('Continuous Retraining Pipeline', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B))),
            ],
          ),
          const SizedBox(height: 8),
          const Text('End-to-end automated pipeline to retrain, evaluate, and deploy deepfake detection models.',
            style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
          ),
          const SizedBox(height: 28),

          // Stage 1: Retraining
          _stageHeader(1, 'Model Retraining', _currentStage >= 1),
          if (_currentStage >= 1) _buildRetrainingStage(),

          const SizedBox(height: 32),

          // Stage 2: Evaluation
          _stageHeader(2, 'Candidate Evaluation', _currentStage >= 2),
          if (_currentStage >= 2) _buildEvaluationStage(),

          const SizedBox(height: 32),

          // Stage 3: Deployment
          _stageHeader(3, 'Deployment Approval', _currentStage >= 3),
          if (_currentStage >= 3) _buildDeploymentStage(),
          
          const SizedBox(height: 48),
        ],
      ),
    );
  }

  Widget _stageHeader(int stageNumber, String title, bool isActive) {
    final color = isActive ? const Color(0xFF6366F1) : const Color(0xFF94A3B8);
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        children: [
          Container(
            width: 28, height: 28,
            decoration: BoxDecoration(
              color: isActive ? color : color.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            alignment: Alignment.center,
            child: Text('$stageNumber', style: TextStyle(color: isActive ? Colors.white : color, fontWeight: FontWeight.w800, fontSize: 13)),
          ),
          const SizedBox(width: 12),
          Text(title, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: isActive ? const Color(0xFF1E293B) : const Color(0xFF94A3B8))),
          if (isActive && stageNumber < _currentStage) ...[
            const SizedBox(width: 12),
            const Icon(Icons.check_circle_rounded, color: Color(0xFF22C55E), size: 18),
          ]
        ],
      ),
    );
  }

  Widget _buildRetrainingStage() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_currentStage == 1 && !_isTraining && _currentStepIndex < 0) ...[
            const Text('Target Model', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
            const SizedBox(height: 12),
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                childAspectRatio: 4.5,
              ),
              itemCount: mockModelCards.length,
              itemBuilder: (context, index) {
                final model = mockModelCards[index];
                final isSelected = _selectedModel == model.name;
                
                return InkWell(
                  onTap: _isTraining ? null : () => setState(() => _selectedModel = model.name),
                  borderRadius: BorderRadius.circular(12),
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: isSelected ? const Color(0xFF6366F1).withValues(alpha: 0.05) : Colors.white.withValues(alpha: 0.6),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isSelected ? const Color(0xFF6366F1) : Colors.black.withValues(alpha: 0.06),
                        width: isSelected ? 2 : 1,
                      ),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(model.name, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
                              const SizedBox(height: 2),
                              Text(model.description, style: const TextStyle(fontSize: 12, color: Color(0xFF64748B)), maxLines: 1, overflow: TextOverflow.ellipsis),
                              const SizedBox(height: 6),
                              _miniBadge('v${model.currentVersion}', Colors.grey.shade700),
                            ],
                          ),
                        ),
                        if (isSelected)
                          const Icon(Icons.check_circle_rounded, color: Color(0xFF6366F1), size: 20),
                      ],
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                ElevatedButton.icon(
                  onPressed: _selectedModel == null ? null : _startTraining,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF6366F1),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  icon: const Icon(Icons.play_arrow_rounded, size: 20),
                  label: const Text('Start Retraining'),
                ),
              ],
            ),
          ] else ...[
            // Progress Area
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  _currentStage > 1 ? 'Retraining Completed: $_selectedModel' : 'Retraining: $_selectedModel',
                  style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF1E293B)),
                ),
                if (_isTraining)
                  const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF6366F1))),
              ],
            ),
            const SizedBox(height: 16),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: _currentStepIndex >= 0 ? (_currentStepIndex + 1) / mockTrainingSteps.length : 0,
                backgroundColor: const Color(0xFFF1F5F9),
                valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF6366F1)),
                minHeight: 8,
              ),
            ),
            const SizedBox(height: 24),
            
            // Logs Terminal
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF0F172A),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF334155)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: List.generate(mockTrainingSteps.length, (index) {
                  final step = mockTrainingSteps[index];
                  
                  String statusStr = 'pending';
                  Color statusColor = const Color(0xFF64748B);
                  IconData statusIcon = Icons.hourglass_empty;
                  
                  if (_currentStepIndex > index || _currentStage > 1) {
                    statusStr = 'done';
                    statusColor = const Color(0xFF22C55E);
                    statusIcon = Icons.check_circle_outline;
                  } else if (_currentStepIndex == index && _isTraining) {
                    statusStr = 'running';
                    statusColor = const Color(0xFF3B82F6);
                    statusIcon = Icons.sync;
                  }

                  final isPending = statusStr == 'pending';
                  
                  return Opacity(
                    opacity: isPending ? 0.4 : 1.0,
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 8.0),
                      child: Row(
                        children: [
                          _isTraining && _currentStepIndex == index 
                            ? const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF3B82F6)))
                            : Icon(statusIcon, size: 14, color: statusColor),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              step['step'] as String,
                              style: const TextStyle(color: Colors.white, fontSize: 13, fontFamily: 'monospace'),
                            ),
                          ),
                          Text(
                            (statusStr == 'done' && _currentStepIndex >= index) || _currentStage > 1 ? step['duration'] as String : (statusStr == 'running' ? '...' : ''),
                            style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12, fontFamily: 'monospace'),
                          ),
                        ],
                      ),
                    ),
                  );
                }),
              ),
            ),
          ]
        ],
      ),
    );
  }

  Widget _buildEvaluationStage() {
    return _card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _modelInfoCard('Current Deployed', mockCurrentModelVersion, const Color(0xFF64748B), true),
              const SizedBox(width: 24),
              const Icon(Icons.compare_arrows_rounded, color: Color(0xFF94A3B8)),
              const SizedBox(width: 24),
              _modelInfoCard('Candidate', mockCandidateModelVersion, const Color(0xFF6366F1), false),
            ],
          ),
          const SizedBox(height: 24),
          Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFFE2E8F0)),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Column(
                children: [
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
                  ...mockMetrics.asMap().entries.map((entry) {
                    final idx = entry.key;
                    final m = entry.value;
                    final isLast = idx == mockMetrics.length - 1;
                    
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
          // Proceed to deploy button
          if (_currentStage == 2)
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                ElevatedButton.icon(
                  onPressed: () => setState(() => _currentStage = 3),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF6366F1),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  icon: const Icon(Icons.arrow_forward_rounded, size: 20),
                  label: const Text('Proceed to Deployment'),
                ),
              ],
            )
        ],
      ),
    );
  }

  Widget _buildDeploymentStage() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          flex: 3,
          child: Column(
            children: [
              _card(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Model Version Bump', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        _versionPill(mockCurrentModelVersion, isCurrent: true),
                        const Padding(
                          padding: EdgeInsets.symmetric(horizontal: 16),
                          child: Icon(Icons.arrow_forward_rounded, color: Color(0xFF94A3B8)),
                        ),
                        _versionPill(mockCandidateModelVersion, isCurrent: false),
                      ],
                    ),
                    const SizedBox(height: 24),
                    const Text(
                      'Approving deployment will hot-swap the model in the production API. '
                      'The current model will be archived.',
                      style: TextStyle(fontSize: 13, color: Color(0xFF64748B), height: 1.5),
                    ),
                    const SizedBox(height: 24),
                    
                    if (!_isDeployed && !_isRejected) ...[
                      Row(
                        children: [
                          ElevatedButton.icon(
                            onPressed: _isDeploying ? null : _deploy,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF22C55E),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              elevation: 0,
                            ),
                            icon: _isDeploying 
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                              : const Icon(Icons.rocket_launch, size: 18),
                            label: Text(_isDeploying ? 'Deploying...' : 'Approve & Deploy'),
                          ),
                          const SizedBox(width: 12),
                          OutlinedButton(
                            onPressed: _isDeploying ? null : _reject,
                            style: OutlinedButton.styleFrom(
                              foregroundColor: const Color(0xFFEF4444),
                              side: const BorderSide(color: Color(0xFFEF4444)),
                              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                            child: const Text('Reject'),
                          )
                        ],
                      ),
                    ] else if (_isDeployed) ...[
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF22C55E).withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: const Color(0xFF22C55E).withValues(alpha: 0.2)),
                        ),
                        child: const Row(
                          children: [
                            Icon(Icons.check_circle, color: Color(0xFF22C55E)),
                            SizedBox(width: 12),
                            Text('Deployed to Production API', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF166534))),
                          ],
                        ),
                      ),
                    ] else if (_isRejected) ...[
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFFEF4444).withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: const Color(0xFFEF4444).withValues(alpha: 0.2)),
                        ),
                        child: const Row(
                          children: [
                            Icon(Icons.block, color: Color(0xFFEF4444)),
                            SizedBox(width: 12),
                            Text('Candidate rejected.', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF991B1B))),
                          ],
                        ),
                      ),
                    ]
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(width: 24),
        Expanded(
          flex: 2,
          child: _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Deployment Timeline', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF1E293B))),
                const SizedBox(height: 20),
                ...List.generate(mockDeploymentTimeline.length, (index) {
                  final item = mockDeploymentTimeline[index];
                  final isLastItem = index == mockDeploymentTimeline.length - 1;
                  
                  String status = item['status'] as String;
                  if (isLastItem) {
                    if (_isDeployed) status = 'done';
                    else if (_isRejected) status = 'rejected';
                  }
                  if (index == mockDeploymentTimeline.length - 2) {
                    if (_isDeployed || _isRejected) status = 'done';
                  }

                  final isDone = status == 'done';
                  final isRejectedStatus = status == 'rejected';
                  final isActive = status == 'active';
                  
                  final color = isDone ? const Color(0xFF22C55E) : (isRejectedStatus ? const Color(0xFFEF4444) : (isActive ? const Color(0xFF3B82F6) : const Color(0xFFCBD5E1)));

                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Column(
                        children: [
                          Container(
                            width: 16, height: 16,
                            decoration: BoxDecoration(
                              color: isDone || isRejectedStatus ? color : Colors.white,
                              border: Border.all(color: color, width: 2),
                              shape: BoxShape.circle,
                            ),
                            child: isDone
                              ? const Icon(Icons.check, size: 10, color: Colors.white)
                              : (isRejectedStatus ? const Icon(Icons.close, size: 10, color: Colors.white) : null),
                          ),
                          if (!isLastItem)
                            Container(width: 2, height: 32, color: isActive || isDone ? const Color(0xFF22C55E) : const Color(0xFFE2E8F0)),
                        ],
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              item['event'] as String,
                              style: TextStyle(fontSize: 13, fontWeight: isActive ? FontWeight.w700 : FontWeight.w500, color: isActive ? color : const Color(0xFF334155)),
                            ),
                            const SizedBox(height: 2),
                            if (item['time'] != '—' || (isLastItem && _isDeployed))
                              Text(isLastItem && _isDeployed ? 'Just now' : item['time'] as String, style: const TextStyle(fontSize: 11, color: Color(0xFF94A3B8))),
                          ],
                        ),
                      )
                    ],
                  );
                })
              ],
            ),
          ),
        )
      ],
    );
  }

  Widget _miniBadge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Text(text, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: color)),
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

  Widget _versionPill(String version, {required bool isCurrent}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: isCurrent ? const Color(0xFFF1F5F9) : const Color(0xFF6366F1).withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: isCurrent ? const Color(0xFFE2E8F0) : const Color(0xFF6366F1).withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(isCurrent ? 'Current' : 'Candidate', style: const TextStyle(fontSize: 11, color: Color(0xFF64748B))),
          const SizedBox(height: 2),
          Text(version, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: isCurrent ? const Color(0xFF334155) : const Color(0xFF6366F1), fontFamily: 'monospace')),
        ],
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
