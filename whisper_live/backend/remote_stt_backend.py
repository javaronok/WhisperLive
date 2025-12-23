from typing import Optional, Callable, Any, Dict, List
from whisper_live.backend.base import ServeClientBase
from RealtimeSTT import AudioToTextRecorderClient

DEFAULT_RECORDER_CONFIG: Dict[str, Any] = {
    "use_microphone": False,
    "spinner": False,
    "model": "tiny",
    "realtime_model_type": "tiny",
    "use_main_model_for_realtime": False,
    "language": "ru", # Default, will be overridden by source_language in __init__
    "silero_sensitivity": 0.95,
    "webrtc_sensitivity": 3,
    "post_speech_silence_duration": 0.7,
    "min_length_of_recording": 0.5,
    "min_gap_between_recordings": 0,
    "enable_realtime_transcription": True,
    "realtime_processing_pause": 0.03,
    "silero_use_onnx": True,
    "silero_deactivity_detection": True,
    "early_transcription_on_silence": 0,
    "beam_size": 3,
    "beam_size_realtime": 3,
    "no_log_file": True,
    "wake_words": "jarvis",
    "wakeword_backend": "pvporcupine",
    "allowed_latency_limit": 500,
    # Callbacks will be added dynamically in _create_recorder
    "debug_mode": True,
    "initial_prompt_realtime": "The sky is blue. When the sky... She walked home. Because he... Today is sunny. If only I...",
    "faster_whisper_vad_filter": False,
}

class RemoteSTTBackend(ServeClientBase):
    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        translation_queue=None,
        translation_client=None,
        remote_stt_server_host=None,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
            translation_client
        )

        self.recorder = AudioToTextRecorderClient(**active_config)
        # Ensure wake words are disabled if needed (can also be done via config dict)
        self._set_recorder_param("use_wake_words", False)