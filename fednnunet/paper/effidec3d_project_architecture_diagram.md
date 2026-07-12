# FedUnlearning nnU-Net + EffiDec3D Architecture Diagram

## 1. 전체 프로젝트 구조

```mermaid
flowchart TB
    User["User / Experiment Script"]
    Run["fednnunet/run.py<br/>Multi-client orchestration CLI"]

    subgraph Options["Experiment Options"]
        Task["task<br/>plan_and_preprocess / train / unlearn"]
        DecoderOpt["decoder option<br/>--decoder_arch effidec3d_uxnet"]
        UnlearnSwitch["unlearn-only switch<br/>--unlearn_decoder_switch effidec3d_uxnet"]
    end

    subgraph ServerSide["Server Side"]
        Server["fednnunet/server.py"]
        Strategy["MyStrategy<br/>FedAvg + FedEraser + planning-aware unlearning"]
        PlanDiff["Plan Diff / Policy<br/>P_all vs P_minus_target"]
        Artifacts["Artifacts<br/>fingerprints / retained updates / checkpoints / reports"]
    end

    subgraph ClientSide["Client Side"]
        Client["fednnunet/client.py"]
        FlowerClient["FlowerClient<br/>fednnunet/client_core.py"]
        TrainingContext["ensure_training_context()<br/>select plans + trainer + model"]
    end

    subgraph NnUNetLayer["nnU-Net v2 Layer"]
        Plans["nnU-Net plans<br/>architecture metadata"]
        BaseTrainer["default nnUNetTrainer"]
        EffiTrainer["nnUNetTrainerEffiDec3DUXNet<br/>deep supervision disabled"]
        DefaultUNet["Default nnU-Net architecture"]
        EffiModel["EffiDec3DUXNet<br/>3DUXNET_EffiDec3D adapter"]
    end

    subgraph DataLayer["Data / Model Storage"]
        Raw["nnUNet_raw"]
        Preprocessed["nnUNet_preprocessed"]
        Results["nnUNet_results"]
        FederaserStore["FedEraser history<br/>retained client updates"]
    end

    User --> Run
    Run --> Options
    Run --> Server
    Run --> Client

    Server --> Strategy
    Strategy --> PlanDiff
    Strategy --> Artifacts
    Strategy <--> FlowerClient

    Client --> FlowerClient
    FlowerClient --> TrainingContext
    TrainingContext --> Plans
    TrainingContext --> BaseTrainer
    TrainingContext --> EffiTrainer

    Plans --> DefaultUNet
    Plans --> EffiModel
    BaseTrainer --> DefaultUNet
    EffiTrainer --> EffiModel

    Raw --> Preprocessed
    Preprocessed --> Plans
    Preprocessed --> TrainingContext
    TrainingContext --> Results
    Strategy --> FederaserStore
    FederaserStore --> Strategy
```

## 2. EffiDec3D 모델 내부 구조

```mermaid
flowchart LR
    X["Input 3D volume<br/>B x C x D x H x W"]

    subgraph Encoder["UXNet Encoder"]
        E1["Stage 1<br/>1/2 resolution<br/>feat_size[0]"]
        E2["Stage 2<br/>1/4 resolution<br/>feat_size[1]"]
        E3["Stage 3<br/>1/8 resolution<br/>feat_size[2]"]
        E4["Stage 4<br/>1/16 resolution<br/>feat_size[3]"]
    end

    subgraph Decoder["EffiDec3D Decoder"]
        C["Channel Reduction Strategy<br/>fixed decoder width:<br/>--effidec_n_decoder_channels"]
        R["High-Resolution Layer Removal<br/>controlled by:<br/>--effidec_resolution_factor"]
        D5["Decoder from 1/16"]
        D4["Decoder from 1/8"]
        D3["Decoder from 1/4"]
        D2["Optional decoder from 1/2"]
        D1["Optional full-resolution decoder"]
    end

    Head["Segmentation Head<br/>UnetOutBlock"]
    Up["Compatibility Upsampling<br/>logits resized to nnU-Net target shape"]
    Y["Output logits<br/>B x classes x D x H x W"]

    X --> E1 --> E2 --> E3 --> E4
    E4 --> D5
    E3 -.skip.-> D5
    D5 --> D4
    E2 -.skip.-> D4
    D4 --> D3
    E1 -.skip.-> D3
    D3 --> D2 --> D1

    C --> D5
    C --> D4
    C --> D3
    R --> D2
    R --> D1

    D3 --> Head
    D2 --> Head
    D1 --> Head
    Head --> Up --> Y
```

Default setting in this update:

```text
--effidec_n_decoder_channels 48
--effidec_resolution_factor 2
```

This means the default EffiDec3D path uses both:

- channel reduction in the decoder
- removal of the full-resolution decoder stage, followed by output upsampling for nnU-Net compatibility

## 3. Train + Unlearn With EffiDec3D

```mermaid
sequenceDiagram
    participant U as User
    participant R as fednnunet/run.py
    participant S as Server MyStrategy
    participant C as Retained Clients
    participant P as nnU-Net Plans
    participant M as EffiDec3DUXNet
    participant A as Artifacts

    U->>R: train --decoder_arch effidec3d_uxnet
    R->>C: start clients with decoder options
    C->>P: create derived EffiDec3D plans if needed
    C->>M: instantiate nnUNetTrainerEffiDec3DUXNet
    S->>C: send global parameters
    C->>S: return local model updates
    S->>S: FedAvg aggregation
    S->>A: save retained updates for FedEraser

    U->>R: unlearn --decoder_arch effidec3d_uxnet --target_client ku
    R->>S: start unlearning with EffiDec3D plans
    S->>S: compute P_all vs P_minus_target
    S->>C: exclude target client
    S->>C: calibration / Level 2 retraining as selected
    C->>M: train retained clients with EffiDec3D model
    C->>S: return retained-client updates
    S->>A: save unlearned checkpoint and report
```

## 4. Experimental Unlearn-Only Switch

```mermaid
sequenceDiagram
    participant U as User
    participant R as fednnunet/run.py
    participant S as Server MyStrategy
    participant C as Retained Clients
    participant N as Original nnU-Net Model
    participant E as EffiDec3DUXNet
    participant A as Report

    U->>R: unlearn --unlearn_decoder_switch effidec3d_uxnet
    R->>S: request unlearn-only architecture switch
    S->>S: force Level 2 unlearning path
    S->>S: generate EffiDec3D Level 2 plans
    S->>C: send original/global checkpoint
    C->>E: instantiate EffiDec3DUXNet
    C->>E: partial compatible weight transfer from original model
    C->>S: return transfer report + retrained model
    S->>A: record transfer_ratio, skipped keys, final checkpoint
```

## 5. 발표용 핵심 메시지

```text
The update integrates EffiDec3D as a selectable nnU-Net v2 architecture.
The integration is plan-driven, so the federated training and unlearning
pipelines can choose either the original nnU-Net model or the EffiDec3D UXNet
model without changing the core Flower/FedEraser protocol.

For efficiency, the implemented EffiDec3D path uses:
1. Channel Reduction Strategy via --effidec_n_decoder_channels
2. High-Resolution Layer Removal via --effidec_resolution_factor

The default configuration reduces decoder cost while resizing logits back to
the original nnU-Net segmentation target resolution.
```
