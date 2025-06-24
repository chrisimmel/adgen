# Quiksilver Ad with Veo 3

https://github.com/user-attachments/assets/6031deb6-2b99-435a-a60d-bb553ccba67b

This is the first ad generated using Veo 3. It was created with the following command:

```bash
uv run adgen generate -u https://www.quiksilver.fr/
```

Here is the relevant portion of the config.yaml:

```yaml
providers:
  llm: "anthropic" # openai, anthropic
  video: "veo3" # runwayml, pika, veo3, mock

# LLM Settings
llm:
  anthropic:
    model: "claude-sonnet-4-0"
    temperature: 0.7

video:
  aspect_ratio: "16:9"
  quality: "high"
  veo3:
    max_duration: 8
    min_duration: 5
    max_scenes: 1 # Single comprehensive clip
    single_clip_mode: true # Generate entire sequence in one clip
    generate_audio: true # Cheaper without audio
    timeout_seconds: 300
```

Note that video generation using Veo 3 is relatively expensive, so that discourages experimentation. This video cost $6 (8s at $0.75/s). As long as the quality is high and we don't need to
regenerate frequently, that's of course not a high cost at all
for a video ad. It's only when experimenting and developing the
tool that it becomes expensive.

Because Veo 3 can generate sound with the video, this ad doesn't use any music or audio generation other than what is native in the video model. It would be useful to add music and/or narration to help carry the message.
