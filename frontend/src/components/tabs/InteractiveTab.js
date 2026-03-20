import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Field, TextInput } from '../ui/Field';
import { SegTab } from './SegTab';
const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json';
const DEFAULT_CKPT = 'ckpt/interactive_seg.ckpt';
export function InteractiveTab({ glbPath }) {
    const [glb, setGlb] = useState(glbPath ?? '');
    const [ckpt, setCkpt] = useState(DEFAULT_CKPT);
    const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS);
    const [img, setImg] = useState('');
    const [points, setPoints] = useState('388 448 392');
    useEffect(() => { setGlb(glbPath ?? ''); }, [glbPath]);
    return (_jsx(SegTab, { title: "Interactive Part Segmentation", description: "Specify a 3D voxel coordinate (0\u2013511 grid) to isolate a specific part.", runEndpoint: "/api/jobs/interactive", runLabel: "Run Interactive Segmentation", buildParams: (sampler) => ({
            glb_path: glb,
            ckpt_path: ckpt,
            transforms_path: transforms,
            rendered_img: img || null,
            points_str: points,
            ...sampler,
        }), extraInputs: _jsxs(_Fragment, { children: [_jsx(Field, { label: "GLB path", children: _jsx(TextInput, { value: glb, onChange: e => setGlb(e.target.value), placeholder: "Leave empty to use uploaded model" }) }), _jsx(Field, { label: "Checkpoint (.ckpt)", children: _jsx(TextInput, { value: ckpt, onChange: e => setCkpt(e.target.value) }) }), _jsx(Field, { label: "Transforms JSON", children: _jsx(TextInput, { value: transforms, onChange: e => setTransforms(e.target.value) }) }), _jsx(Field, { label: "Override rendered image (optional)", children: _jsx(TextInput, { value: img, onChange: e => setImg(e.target.value), placeholder: "path/to/image.png" }) }), _jsx(Field, { label: "Voxel click points (x y z, up to 10)", children: _jsx(TextInput, { value: points, onChange: e => setPoints(e.target.value), placeholder: "388 448 392   256 256 256" }) })] }) }));
}
