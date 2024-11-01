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
