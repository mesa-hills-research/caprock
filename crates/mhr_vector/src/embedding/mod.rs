//! The arko_index integration surface (unit U20; REQUIREMENTS.md, "Embedding models and
//! the cache").
//!
//! - [`EmbeddingModel`] describes where an index's vectors come from. It is saved in the
//!   index configuration; opening an index for a model whose document vectors differ
//!   ([`EmbeddingModel::requires_reembed`]) fails with `ModelMismatch`, and the caller
//!   re-embeds into a new index.
//! - [`EmbeddingCache`] keeps vectors by (model, content hash), so a chunk whose rendered
//!   text did not change is never embedded twice, across re-indexing and across indexes.
//!
//! This crate never calls a provider: the caller (arko_index, through arko_ai's Voyage
//! client) embeds, and hands vectors to the cache and the store.

mod cache;

pub use cache::{CacheStats, EmbeddingCache};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// SHA-256 (or any 32-byte digest the caller fixes) of the exact text sent to the model
/// (arko_index's rendered chunk: header plus source). The caller computes it.
pub type ContentHash = [u8; 32];

/// Which side of retrieval a text is embedded for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    Document,
    Query,
}

/// Where an index's vectors come from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingModel {
    /// `voyage` (later also local providers).
    pub provider: String,
    /// The provider's model name, e.g. `voyage-code-4`.
    pub model: String,
    /// Output dimension (voyage-code-4: 256, 512, 1024 (default) or 2048).
    pub dimensions: u32,
    /// The provider's input type for documents (`document` for Voyage), if it has one.
    #[serde(default)]
    pub document_input: Option<String>,
    /// The provider's input type for queries (`query` for Voyage), if it has one.
    #[serde(default)]
    pub query_input: Option<String>,
    /// The requested output type (`float`; the store quantizes itself).
    pub output_dtype: String,
    /// Whether the provider returns unit-length vectors.
    pub normalized: bool,
    /// The caller's name for how chunks are rendered into text (e.g.
    /// `arko_index/chunk-v1`: path, a newline, the source). Changing the rendering
    /// changes every vector, so it is part of the model's identity.
    pub text_format: String,
    /// A provider-side model revision, when the provider exposes one.
    #[serde(default)]
    pub revision: Option<String>,
}

impl EmbeddingModel {
    /// Voyage's `voyage-code-4` at `dimensions` (256, 512, 1024 or 2048), document and
    /// query input types, float output, unit-length vectors, `arko_index/chunk-v1` text.
    pub fn voyage_code_4(dimensions: u32) -> EmbeddingModel {
        EmbeddingModel {
            provider: "voyage".into(),
            model: "voyage-code-4".into(),
            dimensions,
            document_input: Some("document".into()),
            query_input: Some("query".into()),
            output_dtype: "float".into(),
            normalized: true,
            text_format: "arko_index/chunk-v1".into(),
            revision: None,
        }
    }

    /// The input type to send for `input`, if the provider has input types.
    pub fn input_type(&self, input: InputType) -> Option<&str> {
        match input {
            InputType::Document => self.document_input.as_deref(),
            InputType::Query => self.query_input.as_deref(),
        }
    }

    /// Whether document vectors from `other` differ from this model's, so an index of
    /// this model cannot take them: any of provider, model, dimensions, document input
    /// type, output type, text format or revision differs. (A different query input
    /// type, or `normalized`, does not change stored vectors.)
    pub fn requires_reembed(&self, other: &EmbeddingModel) -> bool {
        todo!("U20 embedding: EmbeddingModel::requires_reembed")
    }

    /// A stable, file-name-safe identity of the document vectors:
    /// `<provider>-<model>-<dimensions>-<dtype>-<document input or "none">-<text format>`
    /// (plus `-<revision>` when set), every byte outside `[A-Za-z0-9._-]` replaced by
    /// `_`. Two models have equal fingerprints exactly when neither
    /// [`requires_reembed`](Self::requires_reembed) the other (unless replaced bytes
    /// collide, which callers avoid by using plain names).
    pub fn document_fingerprint(&self) -> String {
        todo!("U20 embedding: EmbeddingModel::document_fingerprint")
    }

    /// Checks the fields: non-empty provider, model, dtype and text format (at most 128
    /// bytes each), dimensions in `1..=MAX_DIM`. `InvalidConfig` otherwise.
    pub fn validate(&self) -> Result<()> {
        todo!("U20 embedding: EmbeddingModel::validate")
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} ({} dimensions, {})",
            self.provider, self.model, self.dimensions, self.text_format
        )
    }
}
