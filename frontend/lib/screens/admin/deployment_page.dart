import 'package:flutter/material.dart';
import 'mock_data.dart';

class DeploymentPage extends StatefulWidget {
  const DeploymentPage({super.key});

  @override
  State<DeploymentPage> createState() => _DeploymentPageState();
}

class _DeploymentPageState extends State<DeploymentPage> {
  bool _isDeploying = false;
  bool _isDeployed = false;
  bool _isRejected = false;

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
              Icon(Icons.rocket_launch_outlined, color: Color(0xFF6366F1), size: 22),
              SizedBox(width: 10),
              Text('Deployment Approval', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF1E293B))),
            ],
          ),
          const SizedBox(height: 8),
          const Text('Review and approve the candidate model for production deployment.',
            style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
          ),
          const SizedBox(height: 28),

          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Left: Info & Actions
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
                            'The current model will be archived and can be rolled back if necessary.',
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
                                  child: const Text('Reject Candidate'),
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
                                  Text('Candidate model rejected.', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Color(0xFF991B1B))),
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
              // Right: Timeline
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
                        
                        // Override final status based on UI actions
                        String status = item['status'] as String;
                        if (isLastItem) {
                          if (_isDeployed) status = 'done';
                          else if (_isRejected) status = 'rejected';
                        }
                        
                        // For the approval wait step
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
                                    style: TextStyle(
                                      fontSize: 13,
                                      fontWeight: isActive ? FontWeight.w700 : FontWeight.w500,
                                      color: isActive ? color : const Color(0xFF334155),
                                    ),
                                  ),
                                  const SizedBox(height: 2),
                                  if (item['time'] != '—' || (isLastItem && _isDeployed))
                                    Text(
                                      isLastItem && _isDeployed ? 'Just now' : item['time'] as String,
                                      style: const TextStyle(fontSize: 11, color: Color(0xFF94A3B8)),
                                    ),
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
          )
        ],
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
}
