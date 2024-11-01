library(biomaRt)
library(dplyr)
library(anndata)


extract_gene_ids <- function(filename) {
  # Read the GTF file
  gtf_data <- read.table(filename, sep="\t", header=FALSE, stringsAsFactors=FALSE, comment.char="#")
  
  # Extract the 9th column which contains the attribute field
  attributes <- gtf_data[, 9]
  
  # Extract gene_id from each attribute field
  gene_ids <- sapply(attributes, function(x) {
    fields <- strsplit(x, ";[[:space:]]*")[[1]]
    gene_id_field <- fields[grep("^gene_id", fields)]
    if (length(gene_id_field) > 0) {
      gene_id <- sub('gene_id ', '', gene_id_field)
      gene_id <- sub('"$', '', gene_id)
      return(gene_id)
    } else {
      return(NA)
    }
  })
  
  # Remove NA values and get unique gene_ids
  gene_ids <- unique(na.omit(gene_ids))
  
  # Return the gene_ids as a list
  return(gene_ids)
}

ensembl <- useEnsembl(biomart = "genes")
ensembl_human <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

get_genes_homology <- function(specie, non_human) {
  ensembl_nonhuman <- useEnsembl(biomart = "genes", dataset = paste0(specie, "_gene_ensembl"),  host="https://jan2024.archive.ensembl.org")

  gene_mapping <- getBM(
    attributes = c("ensembl_gene_id", "external_gene_name",
                  "hsapiens_homolog_ensembl_gene", "hsapiens_homolog_associated_gene_name"),
    filters = "ensembl_gene_id",
    values = non_human,
    mart = ensembl_nonhuman
  )

  gene_mapping <- gene_mapping[gene_mapping$hsapiens_homolog_ensembl_gene != "",]
  return(gene_mapping)
}

species <- c("drerio", "ggallus", "mmusculus", "rnorvegicus", "sscrofa")
for (specie in species) {
  non_human <- extract_gene_ids(paste0(specie, ".gtf"))
  table <- get_genes_homology(specie, non_human)
  if (is.null(new_table)) {
    new_table <- table
    } else {
    new_table <- merge(new_table, table, by="hsapiens_homolog_ensembl_gene")
  }
}
write.table(x=new_table, file="Hsapiens_features.txt")

# Set datasets for pig and human


