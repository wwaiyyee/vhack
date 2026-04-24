// frontend/lib/screens/admin/mock_data.dart
/// Centralized mock data for the retraining pipeline admin dashboard.
/// Replace with real API calls when backend integration is ready.

class MockSample {
  final String id;
  final String fileType;     // image, video, audio
  final String fileName;
  final String uploadTime;
  final String groundTruth;   // True or False
  final String status;        // pending, approved, rejected

  const MockSample({
    required this.id,
    required this.fileType,
    required this.fileName,
    required this.uploadTime,
    required this.groundTruth,
    required this.status,
  });
}

class MockModelCard {
  final String name;
  final String description;
  final String icon;
  final String currentVersion;
  final int trainingSamples;

  const MockModelCard({
    required this.name,
    required this.description,
    required this.icon,
    required this.currentVersion,
    required this.trainingSamples,
  });
}

class MockMetric {
  final String name;
  final double currentValue;
  final double candidateValue;

  const MockMetric({
    required this.name,
    required this.currentValue,
    required this.candidateValue,
  });
}

// ─── Retained Samples ────────────────────────────────────────────────────
final List<MockSample> mockSamples = [
  const MockSample(id: 'SAM-001', fileType: 'image', fileName: 'face_swap_001.jpg', uploadTime: '2026-04-24 09:12:33', groundTruth: 'False', status: 'approved'),
  const MockSample(id: 'SAM-002', fileType: 'video', fileName: 'deepfake_clip.mp4', uploadTime: '2026-04-24 08:45:10', groundTruth: 'False', status: 'approved'),
  const MockSample(id: 'SAM-003', fileType: 'audio', fileName: 'scam_call_recording.wav', uploadTime: '2026-04-24 08:30:55', groundTruth: 'False', status: 'pending'),
  const MockSample(id: 'SAM-004', fileType: 'image', fileName: 'real_selfie_004.png', uploadTime: '2026-04-24 07:58:22', groundTruth: '?', status: 'pending'),
  const MockSample(id: 'SAM-005', fileType: 'video', fileName: 'lip_sync_fake.mp4', uploadTime: '2026-04-24 07:22:11', groundTruth: 'False', status: 'approved'),
  const MockSample(id: 'SAM-006', fileType: 'audio', fileName: 'voice_clone_test.mp3', uploadTime: '2026-04-24 06:15:44', groundTruth: 'False', status: 'rejected'),
  const MockSample(id: 'SAM-007', fileType: 'image', fileName: 'gan_generated_face.jpg', uploadTime: '2026-04-24 05:40:00', groundTruth: 'False', status: 'approved'),
  const MockSample(id: 'SAM-008', fileType: 'video', fileName: 'real_video_interview.mp4', uploadTime: '2026-04-23 22:10:38', groundTruth: 'True', status: 'approved'),
  const MockSample(id: 'SAM-009', fileType: 'audio', fileName: 'bank_impersonation.wav', uploadTime: '2026-04-23 21:55:12', groundTruth: 'False', status: 'rejected'),
  const MockSample(id: 'SAM-010', fileType: 'image', fileName: 'stable_diffusion_out.png', uploadTime: '2026-04-23 20:32:05', groundTruth: 'False', status: 'pending'),
  const MockSample(id: 'SAM-011', fileType: 'video', fileName: 'faceswap_zoom_call.mp4', uploadTime: '2026-04-23 19:15:33', groundTruth: 'False', status: 'approved'),
  const MockSample(id: 'SAM-012', fileType: 'audio', fileName: 'otp_phishing_call.wav', uploadTime: '2026-04-23 18:44:22', groundTruth: 'False', status: 'approved'),
];

// ─── Dataset Builder ─────────────────────────────────────────────────────
const int mockApprovedTotal = 8;
const int mockApprovedImages = 3;
const int mockApprovedVideos = 3;
const int mockApprovedAudios = 2;
const String mockDatasetVersion = 'ds_2026_04_24_v1';

// ─── Model Cards ─────────────────────────────────────────────────────────
final List<MockModelCard> mockModelCards = [
  const MockModelCard(name: 'FaceForge XceptionNet', description: 'Primary spatial detector (90% FF++ C23)', icon: '🖼️', currentVersion: 'v2.3.1', trainingSamples: 24500),
  const MockModelCard(name: 'Fine-tuned ViT TwoStage', description: 'Challenger vision transformer model', icon: '🖼️', currentVersion: 'v1.8.0', trainingSamples: 8200),
  const MockModelCard(name: 'EfficientNet CelebDF', description: 'Fallback tiebreaker model', icon: '🖼️', currentVersion: 'v1.1.4', trainingSamples: 12000),
  const MockModelCard(name: 'XLS-R Deepfake', description: 'Primary audio sequence detector (ASVspoof)', icon: '🎙️', currentVersion: 'v1.5.2', trainingSamples: 15300),
  const MockModelCard(name: 'Wav2Vec2 Deepfake', description: 'Secondary audio detector ensemble', icon: '🎙️', currentVersion: 'v1.2.0', trainingSamples: 6800),
];

// ─── Training Steps ──────────────────────────────────────────────────────
final List<Map<String, dynamic>> mockTrainingSteps = [
  {'step': 'Loading dataset', 'status': 'done', 'duration': '10.0s'},
  {'step': 'Preprocessing & augmentation', 'status': 'done', 'duration': '12.3s'},
  {'step': 'Training (epoch 1/10)', 'status': 'running', 'duration': '—'},
  {'step': 'Training (epoch 2/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 3/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 4/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 5/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 6/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 7/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 8/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 9/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Training (epoch 10/10)', 'status': 'pending', 'duration': '—'},
  {'step': 'Validating on holdout set', 'status': 'pending', 'duration': '—'},
  {'step': 'Saving candidate model', 'status': 'pending', 'duration': '—'},
];

// ─── Evaluation Metrics ──────────────────────────────────────────────────
final List<MockMetric> mockMetrics = [
  const MockMetric(name: 'Accuracy', currentValue: 0.923, candidateValue: 0.941),
  const MockMetric(name: 'Precision', currentValue: 0.918, candidateValue: 0.935),
  const MockMetric(name: 'Recall', currentValue: 0.897, candidateValue: 0.928),
  const MockMetric(name: 'F1-Score', currentValue: 0.907, candidateValue: 0.931),
  const MockMetric(name: 'False Positive Rate', currentValue: 0.082, candidateValue: 0.065),
  const MockMetric(name: 'False Negative Rate', currentValue: 0.103, candidateValue: 0.072),
];

// ─── Deployment ──────────────────────────────────────────────────────────
const String mockCurrentModelVersion = 'v2.3.1';
const String mockCandidateModelVersion = 'v2.4.0-rc1';

final List<Map<String, String>> mockDeploymentTimeline = [
  {'event': 'Candidate model trained', 'time': '2026-04-24 14:22:00', 'status': 'done'},
  {'event': 'Evaluation passed', 'time': '2026-04-24 14:25:33', 'status': 'done'},
  {'event': 'Candidate ready for deployment', 'time': '2026-04-24 14:26:00', 'status': 'done'},
  {'event': 'Waiting for admin approval', 'time': '—', 'status': 'active'},
  {'event': 'Deploy to production', 'time': '—', 'status': 'pending'},
];
