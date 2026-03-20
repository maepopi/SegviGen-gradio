import { useState, useEffect } from 'react'
import { Field, TextInput } from '../ui/Field'
import { SegTab } from './SegTab'
import type { SamplerParams } from '../SamplerFields'

const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json'
const DEFAULT_CKPT       = 'ckpt/interactive_seg.ckpt'

interface Props { glbPath?: string | null }

export function InteractiveTab({ glbPath }: Props) {
  const [glb,        setGlb]        = useState(glbPath ?? '')
  const [ckpt,       setCkpt]       = useState(DEFAULT_CKPT)
  const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS)
  const [img,        setImg]        = useState('')
  const [points,     setPoints]     = useState('388 448 392')

  useEffect(() => { setGlb(glbPath ?? '') }, [glbPath])

  return (
    <SegTab
      title="Interactive Part Segmentation"
      description="Specify a 3D voxel coordinate (0–511 grid) to isolate a specific part."
      runEndpoint="/api/jobs/interactive"
      runLabel="Run Interactive Segmentation"
      buildParams={(sampler: SamplerParams) => ({
        glb_path:        glb,
        ckpt_path:       ckpt,
        transforms_path: transforms,
        rendered_img:    img || null,
        points_str:      points,
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
          <Field label="Voxel click points (x y z, up to 10)">
            <TextInput value={points} onChange={e => setPoints(e.target.value)}
              placeholder="388 448 392   256 256 256" />
          </Field>
        </>
      }
    />
  )
}
