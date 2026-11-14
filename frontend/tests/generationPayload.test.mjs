import assert from 'node:assert/strict';
import test from 'node:test';
import { buildGenerationPayload, validateGeneration } from '../src/views/home/generationPayload.ts';

function baseGeneration(overrides = {}) {
  return {
    activity: 'txt2img', img2imgMode: 'image', preset: 'SD', workflowId: '',
    modelPath: 'checkpoints/demo.safetensors', modelName: 'demo', prompt: 'a lantern',
    negativePrompt: 'blurry', width: 1024, height: 576, steps: 25, cfg: 7,
    sampler: 'euler_ancestral', scheduler: 'karras', seed: 12, denoise: 0.75,
    vaeOverrideEnabled: false, videoVae: 'Automatic', textEncoderOverrideEnabled: false,
    textEncoder: 'Automatic', distilledLoraEnabled: false, musicCaption: '',
    musicMode: 'instrumental', musicLyrics: '', musicNegativePrompt: '',
    musicDurationSeconds: 12, musicCfgScale: 3, musicTopK: 50, musicTiledDecode: true,
    musicFormat: 'flac', resizeMode: 'Just Resize', maskBlur: 8, maskContent: 'Original',
    inpaintEngine: 'lanpaint', differentialStrength: 1, lanpaintThinkingSteps: 5,
    referenceImage: null, extraReferenceImages: [], inpaintMaskImage: null,
    controlNetEnabled: false, controlNetType: 'canny', controlNetModel: 'Automatic',
    controlNetModelName: 'Automatic', controlNetImage: null, controlNetStrength: 0.75,
    controlNetStart: 0, controlNetEnd: 1, controlNetBalance: 'Balanced',
    promptRegions: [], videoFrames: 81, videoFps: 16, videoSeconds: 5,
    videoMotionStrength: 1, videoActiveAudio: false, audioVaeEnabled: false,
    audioVae: '', latentUpscaleEnabled: false, latentUpscale: '', latentUpscaleRefine: false,
    lightningLoraEnabled: false, decodeTilesX: 2, decodeTilesY: 2, decodeOverlap: 16,
    directorEnabled: false, directorDuration: 5, directorLocalPrompt: '',
    directorLocalNegative: '', directorGuideStrength: 0.6, directorUseCustomAudio: false,
    directorResizeMethod: 'bilinear', directorDivisibleBy: 32,
    directorImgCompression: 20, ...overrides,
  };
}
function ctx(overrides = {}, flags = {}) {
  return {
    generation: baseGeneration(overrides),
    activeLoras: [{ name: 'detail', relative_name: 'detail.safetensors', strength: 0.5, strength_model: 0.5, strength_clip: 0.5 }],
    allReferenceImages: overrides.referenceImage ? [overrides.referenceImage, 'data:image/png;base64,EXTRA'] : [],
    alignedFrames: 81, directorFrames: 81,
    controlNetCompatible: ['SD', 'SDXL', 'LTX', 'Wan', 'Flux', 'Qwen'].includes(overrides.preset || 'SD'),
    directorMode: overrides.preset === 'LTX' && !!overrides.directorEnabled,
    ideogram4Mode: overrides.preset === 'Ideogram4',
    videoMode: ['LTX', 'Wan'].includes(overrides.preset || 'SD'),
    music3Mode: overrides.preset === 'Music3',
    music3RecommendedProfile: undefined,
    ...flags,
  };
}

test('plain txt2img retains the backend request contract and LoRAs', () => {
  const p = buildGenerationPayload(ctx());
  assert.equal(p.workspace, 'viewer');
  assert.equal(p.activity, 'txt2img');
  assert.equal(p.workflow_id, null);
  assert.equal(p.model_path, 'checkpoints/demo.safetensors');
  assert.deepEqual(p.loras.map(x => x.name), ['detail']);
  assert.equal(p.denoise, 1);
  assert.deepEqual(p.music, {});
  assert.deepEqual(p.video, {});
  assert.deepEqual(p.director, {});
  assert.deepEqual(p.runtime, {});
});

test('inpainting retains mask, reference, controlnet and differential flags', () => {
  const p = buildGenerationPayload(ctx({
    activity: 'img2img', img2imgMode: 'inpaint', referenceImage: 'data:image/png;base64,REF',
    inpaintMaskImage: 'data:image/png;base64,MASK', inpaintEngine: 'differential',
    controlNetEnabled: true, controlNetModel: 'foo.safetensors',
  }));
  assert.equal(p.denoise, 0.75);
  assert.equal(p.img2img.mode, 'Inpaint masked area');
  assert.equal(p.img2img.mask_image, 'data:image/png;base64,MASK');
  assert.equal(p.img2img.differential_diffusion, true);
  assert.deepEqual(p.img2img.reference_images, ['data:image/png;base64,REF', 'data:image/png;base64,EXTRA']);
  assert.equal(p.controlnet.enabled, true);
  assert.equal(p.controlnet.image, 'data:image/png;base64,REF');
});

test('WAN motion payload keeps aligned fps and audio flags', () => {
  const p = buildGenerationPayload(ctx({ preset: 'Wan', videoActiveAudio: true }, { alignedFrames: 85 }));
  assert.equal(p.video.motion_adapter, 'WAN T2V / I2V');
  assert.equal(p.video.frames, 85);
  assert.equal(p.video.fps, 16);
  assert.equal(p.video.active_audio, true);
  assert.equal(p.video.director_timeline, false);
});

test('LTX director changes workspace and mirrors reference segment metadata', () => {
  const p = buildGenerationPayload(ctx({
    preset: 'LTX', directorEnabled: true,
    referenceImage: 'data:image/png;base64,REF',
  }, { directorFrames: 89 }));
  assert.equal(p.template, 'LTX_DIRECTOR_SUITE');
  assert.equal(p.activity, 'txt2img');
  assert.equal(p.workspace, 'director');
  assert.equal(p.director.duration_frames, 88);
  assert.equal(p.director.timeline_data.references.length, 2);
  assert.deepEqual(JSON.parse(p.director.timeline_data_json), p.director.timeline_data);
  assert.equal(p.video.director_timeline, true);
  assert.deepEqual(p.distilled_loras, [{ name: 'None', strength: 0 }]);
});

test('Ideogram 4 preserves regional layout and shared runtime policy', () => {
  const regions = [{ id: 'a', type: 'text', x: 0.1, y: 0.2, w: 0.3, h: 0.1, prompt: 'title' }];
  const p = buildGenerationPayload(ctx({ preset: 'Ideogram4', promptRegions: regions }));
  assert.equal(p.video.ideogram_reference_mode, 'txt2img');
  assert.deepEqual(p.video.ideogram_regions, regions);
  assert.equal(p.runtime.vram_policy, 'shared');
});

test('Music3 chooses runtime profile from fresh dependency status', () => {
  const state = ctx({ preset: 'Music3', musicMode: 'lyrics', musicLyrics: 'original lyrics' });
  const low = buildGenerationPayload(state);
  const high = buildGenerationPayload({ ...state, music3RecommendedProfile: 'rtx_5090' });
  assert.equal(low.runtime.precision, 'int8');
  assert.equal(high.runtime.precision, 'fp16');
  assert.equal(low.music.lyrics, 'original lyrics');
  assert.equal(high.music.format, 'flac');
});

test('generation validation preserves existing blocking rules and accepts valid dispatch', () => {
  function error(gen, settings = {}) {
    const context = ctx(gen, settings);
    return validateGeneration({
      generation: context.generation,
      payload: buildGenerationPayload(context),
      ideogram4Mode: context.ideogram4Mode,
      videoMode: context.videoMode,
      controlNetCompatible: context.controlNetCompatible,
      controlNetModelCount: settings.controlNetModelCount ?? 1,
    });
  }
  assert.equal(error({ prompt: '   ' }), 'Prompt is required.');
  assert.equal(error({ modelPath: '', modelName: '' }), 'Select a model before generating.');
  assert.equal(error({ activity: 'img2img' }), 'Load a reference image for img2img.');
  assert.equal(error({
    activity: 'img2img', img2imgMode: 'inpaint',
    referenceImage: 'data:image/png;base64,REF',
  }), 'Paint an inpaint mask before generating.');
  assert.match(error({
    preset: 'Ideogram4', activity: 'img2img', img2imgMode: 'inpaint',
    referenceImage: 'data:image/png;base64,REF', inpaintMaskImage: 'data:image/png;base64,MASK',
  }), /does not support true mask inpaint yet/);
  assert.equal(error({
    controlNetEnabled: true, controlNetImage: null,
  }, { controlNetModelCount: 1 }), 'Load a ControlNet image or use the img2img reference.');
  assert.equal(error({
    controlNetEnabled: true, controlNetImage: 'data:image/png;base64,CN',
  }, { controlNetModelCount: 0 }), 'No compatible ControlNet model was detected for this preset and type.');
  assert.equal(error({}), null);
  assert.equal(error({
    activity: 'img2img', img2imgMode: 'inpaint',
    referenceImage: 'data:image/png;base64,REF', inpaintMaskImage: 'data:image/png;base64,MASK',
  }), null);
});
