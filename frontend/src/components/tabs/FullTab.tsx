import { useState, useEffect } from 'react'
import { Field, TextInput } from '../ui/Field'
import { SegTab } from './SegTab'
import type { SamplerParams } from '../SamplerFields'

const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json'
const DEFAULT_CKPT       = 'ckpt/full_seg.ckpt'

interface Props { glbPath?: string | null }

export function FullTab({ glbPath }: Props) {
  const [glb,        setGlb]        = useState(glbPath ?? '')
  const [ckpt,       setCkpt]       = useState(DEFAULT_CKPT)
  const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS)
  const [img,        setImg]        = useState('')

  useEffect(() => { setGlb(glbPath ?? '') }, [glbPath])

  return (
    <SegTab
      title="Full Segmentation"
      description="Automatically segments all parts simultaneously, conditioned on a rendered view of the model."
      runEndpoint="/api/jobs/full"
      runLabel="Run Full Segmentation"
      buildParams={(sampler: SamplerParams) => ({
        glb_path:        glb,
        ckpt_path:       ckpt,
        transforms_path: transforms,
        rendered_img:    img || null,
        ...sampler,
      })}
      extraInputs={
        <>
          <Field label="GLB path">
            <TextInput value={glb} onChange={e => setGlb(e.target.value)}
              placeholder="Leave empty to use uploaded model" />
          </Field>
          <Field label="Checkpoint (.ckpt)">
            <TextInput value={ckpt} onChange={e => setCkpt(e.target.value)} />
          </Field>
          <Field label="Transforms JSON">
            <TextInput value={transforms} onChange={e => setTransforms(e.target.value)} />
          </Field>
          <Field label="Override rendered image (optional)">
            <TextInput value={img} onChange={e => setImg(e.target.value)}
              placeholder="path/to/image.png" />
          </Field>
        </>
      }
    />
  )
}
