        for node in [uploader, downloader]:
            sim.log_info(
                f"{node}: {sum(node.local_blocks)} local blocks, "
                f"{sum(peer is not None for peer in node.backed_up_blocks)} backed up blocks, "
                f"{len(node.remote_blocks_held)} remote blocks held"
            )