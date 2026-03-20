import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Field, TextInput } from '../ui/Field';
import { SegTab } from './SegTab';
const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json';
const DEFAULT_CKPT = 'ckpt/full_seg.ckpt';
export function FullTab({ glbPath }) {
    const [glb, setGlb] = useState(glbPath ?? '');
    const [ckpt, setCkpt] = useState(DEFAULT_CKPT);
    const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS);
    const [img, setImg] = useState('');
    useEffect(() => { setGlb(glbPath ?? ''); }, [glbPath]);
    return (_jsx(SegTab, { title: "Full Segmentation", description: "Automatically segments all parts simultaneously, conditioned on a rendered view of the model.", runEndpoint: "/api/jobs/full", runLabel: "Run Full Segmentation", buildParams: (sampler) => ({
            glb_path: glb,
            ckpt_path: ckpt,
            transforms_path: transforms,
            rendered_img: img || null,
            ...sampler,
        }), extraInputs: _jsxs(_Fragment, { children: [_jsx(Field, { label: "GLB path", children: _jsx(TextInput, { value: glb, onChange: e => setGlb(e.target.value), placeholder: "Leave empty to use uploaded model" }) }), _jsx(Field, { label: "Checkpoint (.ckpt)", children: _jsx(TextInput, { value: ckpt, onChange: e => setCkpt(e.target.value) }) }), _jsx(Field, { label: "Transforms JSON", children: _jsx(TextInput, { value: transforms, onChange: e => setTransforms(e.target.value) }) }), _jsx(Field, { label: "Override rendered image (optional)", children: _jsx(TextInput, { value: img, onChange: e => setImg(e.target.value), placeholder: "path/to/image.png" }) })] }) }));
}
